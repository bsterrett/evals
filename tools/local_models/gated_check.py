"""Acceptance checks for a checkpoint trained to loop, such as a released LoopUS model (WSL).

  parents    compare the checkpoint's weights and tokenizer with candidate parents, on the CPU
  run        check the checkpoint's gated loop and save the full record
  trace      what each pass does, computed by hand: loss, how far the gate opens, confidence
  loss       next-token loss of a plain parent on the same text, for comparison
  summarize  write the compact, tracked summary of a directory of records

`run` checks that:

1. every tensor in the checkpoint file is used and every parameter filled, which
   the loader enforces, and records what the file says about itself;
2. at each fixed number of passes, every forward pass makes the design's layer
   calls and leaves one cache entry per token in every pass's slots;
3. cached decoding agrees with recomputing the whole prefix without a cache;
4. the loop agrees with an explicit recomputation that calls the layers by hand
   and writes the gate's and the confidence head's arithmetic out, with no cache
   and none of the runner's code, both at fixed passes and with stopping, where
   the number of passes used must agree too;
5. a stopping threshold of 1 is identical to the fixed loop, and a threshold of 0
   to one pass;
6. removing the design restores the model exactly.

Checks 2, 5 and 6 are exact in any precision. Checks 3 and 4 compare different
orders of the same arithmetic, so they are decided in FP32 only; in BF16 the
gaps are recorded and not judged.

It also measures, without judging, the next-token loss on a fixed set of
passages at each number of passes and with stopping. `loss` measures the same
for a plain parent. That is a look at whether the rebuilt checkpoint behaves as
its paper describes, on text that happened to be on disk. It validates the tool
and is not a finding about the method.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import LocalRequest, encode_prompt, identity, profiles
from eval_core.local.worker import gpu_lock
from recurrent_retrofit import bench
from recurrent_retrofit.loading import needs_gate
from recurrent_retrofit.program import GatedLoop

PROMPTS = [
    ("raw-capital", "raw-v1", "The capital of France is"),
    ("raw-photosynthesis", "raw-v1", "Photosynthesis is the process by which"),
    ("chat-sum", "native-chat-no-thinking-v1", "What is 2 + 2? Reply with just the number."),
]
FP32_TOLERANCE = 0.01
CONTROL_STEPS = 6
PASSAGE_TASK, PASSAGE_FIELD, PASSAGE_SPLIT = "sciq", "support", "test"
PASSAGE_TOKENS = (32, 256)


def compare(a: list, b: list) -> dict:
    pairs = list(zip(a, b, strict=True))
    return {"steps": len(pairs),
        "identical": all(bool((x == y).all()) for x, y in pairs),
        "argmax_equal": sum(int(x.argmax()) == int(y.argmax()) for x, y in pairs),
        "max_absolute_logit_difference": max(float((x - y).abs().max()) for x, y in pairs)}


def passages(root: Path, tokenizer, count: int) -> tuple[list[list[int]], dict]:
    """The first `count` passages of a pinned benchmark dataset, as token ids. Read offline."""
    import os
    os.environ["HF_HOME"] = str(root.resolve() / "cache/huggingface")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    entry = bench.check_pins(root, [PASSAGE_TASK])[PASSAGE_TASK]
    import datasets
    rows = datasets.load_dataset(entry["dataset"], bench.pin(PASSAGE_TASK).get("config"), revision=entry["revision"],
        split=PASSAGE_SPLIT)
    chosen, seen = [], set()
    for row in rows:
        text = (row[PASSAGE_FIELD] or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if PASSAGE_TOKENS[0] <= len(ids) <= PASSAGE_TOKENS[1]:
            chosen.append(ids)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"Only {len(chosen)} passages fit the length limits")
    digest = hashlib.sha256(json.dumps(chosen).encode()).hexdigest()
    return chosen, {"source": f"{entry['dataset']}@{entry['revision']}", "split": PASSAGE_SPLIT, "field": PASSAGE_FIELD,
        "passages": count, "token_limits": list(PASSAGE_TOKENS), "tokens": sum(map(len, chosen)),
        "scored_tokens": sum(len(ids) - 1 for ids in chosen), "token_ids_sha256": digest}


def passage_loss(model, texts: list[list[int]], after=None) -> dict:
    """Mean next-token loss over every token but each passage's first, one passage per forward pass."""
    import torch
    device = next(model.parameters()).device
    total, scored = 0.0, 0
    with torch.inference_mode():
        for ids in texts:
            row = torch.tensor([ids], device=device)
            logits = model(input_ids=row, use_cache=False).logits[0, :-1].float()
            total += float(torch.nn.functional.cross_entropy(logits, row[0, 1:], reduction="sum"))
            scored += len(ids) - 1
            if after is not None:
                after()
    mean = total / scored
    return {"mean_loss": round(mean, 5), "perplexity": round(math.exp(mean), 3), "scored_tokens": scored}


def check(args) -> bool:
    from recurrent_retrofit.loading import load_parent
    from recurrent_retrofit.reference import ExplicitGated
    from recurrent_retrofit.runner import greedy, install
    import torch
    profile, tokens, root, precision, placement = args.profile, args.tokens, args.root, args.precision, args.placement
    failures: list[str] = []
    on_gpu = placement == "gpu"
    decided = precision == "float32"
    with gpu_lock(root):
        model, tokenizer, info = load_parent(profile, root, precision=precision, placement=placement)
        torch.use_deterministic_algorithms(True)
        layers, unit = info["parent_layers"], tuple(info["gated"]["unit"])
        device = next(model.parameters()).device
        prompts = [(label, encode_prompt(tokenizer, LocalRequest(info["parent"], 1, text, tokens, 0, fmt)))
            for label, fmt, text in PROMPTS]
        first_label, first_ids = prompts[0]
        steps = min(tokens, CONTROL_STEPS)
        explicit = ExplicitGated(model)
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
        texts, text_facts = passages(root, tokenizer, args.passages)

        def bare(ids: list[int]):
            """The checkpoint's layers run once with no design installed. Not a meaningful model; a fixed point for check 6."""
            with torch.inference_mode():
                return model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0].float().cpu()

        def whole(ids: list[int], **kwargs):
            with torch.inference_mode():
                return model(input_ids=torch.tensor([ids], device=device), **kwargs).logits[0].float().cpu()

        def judged(result: dict, label: str, what: str) -> dict:
            result["decided"] = decided
            if decided and result["max_absolute_logit_difference"] > FP32_TOLERANCE:
                failures.append(f"{label}: {what}")
            return result

        def exact(result: dict, label: str, what: str) -> dict:
            if not result["identical"]:
                failures.append(f"{label}: {what}")
            return result

        before = [bare(ids) for _, ids in prompts]
        record = {"profile": profile, "precision": precision, "placement": placement, "tokens_per_prompt": tokens,
            "prompts": [label for label, _ in prompts], "runtime": info,
            "tolerances": {"fp32_max_absolute_logit_difference": FP32_TOLERANCE},
            "passage_text": text_facts, "fixed": [], "stopping": []}
        print(json.dumps({"loaded": info["gated"], "passages": text_facts}), flush=True)

        for passes in args.passes:
            design = GatedLoop(layers, *unit, passes)
            label = f"{design.spec} x{passes}"
            entry = {"design": design.spec, "passes": passes, "layer_calls_per_forward": design.layer_calls}
            try:
                with install(model, design) as installed:
                    rows, outputs = {}, {}
                    for name, ids in prompts:
                        stamps = []
                        def after(step, cache):
                            calls = installed.take_calls()
                            installed.check_calls(calls)
                            installed.check_cache(cache, len(ids) + step)
                            if on_gpu:
                                torch.cuda.synchronize()
                            stamps.append(time.perf_counter())
                        installed.take_calls()
                        if on_gpu:
                            torch.cuda.reset_peak_memory_stats()
                        out = greedy(model, ids, tokens, after_forward=after)
                        outputs[name] = out
                        rows[name] = {"input_tokens": len(ids), "output_token_ids": out["tokens"], "text": decode(out["tokens"]),
                            "decode_tokens_per_second": (len(stamps) - 1) / (stamps[-1] - stamps[0]),
                            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3 if on_gpu else None}
                    cached = outputs[first_label]
                    full = greedy(model, first_ids, steps, cached=False, forced=cached["tokens"])
                    served = whole(first_ids + cached["tokens"][:steps - 1])
                    installed.take_calls()
                    entry["passage_loss"] = passage_loss(model, texts)
                    calls = installed.take_calls()
                    installed.check_calls(calls, len(texts))
                entry["rows"] = rows
                entry["calls_and_cache_verified_every_forward"] = True
                entry["cache_control"] = judged(compare(cached["logits"][:steps], full["logits"]), label,
                    "cached decoding disagrees with the full-prefix control")
                by_hand = explicit.logits(first_ids + cached["tokens"][:steps - 1], design)
                entry["explicit"] = judged(compare(cached["logits"][:steps], list(by_hand[len(first_ids) - 1:])), label,
                    "the runner disagrees with the explicit recomputation")
                entry["explicit_every_position"] = judged(compare([served], [by_hand]), label,
                    "the runner disagrees with the explicit recomputation at some position")
            except RuntimeError as exc:
                entry["error"] = str(exc)
                failures.append(f"{label}: {exc}")
            record["fixed"].append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k != "rows"}), flush=True)
            for name, row in entry.get("rows", {}).items():
                print(f"    {name}: {row['text']!r}", flush=True)

        # The fixed loops that stopping is compared with run without a cache, as stopping does.
        budget = args.budget
        with install(model, GatedLoop(layers, *unit, budget)):
            at_budget = [whole(ids, use_cache=False) for _, ids in prompts]
        with install(model, GatedLoop(layers, *unit, 1)):
            at_one = [whole(ids, use_cache=False) for _, ids in prompts]
        for threshold in (1.0, 0.0, *args.thresholds):
            design = GatedLoop(layers, *unit, budget, threshold)
            label = f"{design.spec} x{budget}"
            entry = {"design": design.spec, "passes": budget, "threshold": threshold, "prompts": {}}
            try:
                with install(model, design) as installed:
                    for index, (name, ids) in enumerate(prompts):
                        installed.take_calls()
                        served = whole(ids)
                        calls = installed.take_calls()
                        installed.check_calls(calls)
                        used = installed.mixer.last_depths[0]
                        row = {"passes_used": used, "layer_calls": sum(calls)}
                        if threshold == 1.0:
                            row["against_the_fixed_loop"] = exact(compare([served], [at_budget[index]]), label,
                                f"a threshold of 1 differs from the fixed loop on {name}")
                        elif threshold == 0.0:
                            row["against_one_pass"] = exact(compare([served], [at_one[index]]), label,
                                f"a threshold of 0 differs from one pass on {name}")
                        entry["prompts"][name] = row
                    installed.take_calls()
                    used_by_passage: list[int] = []
                    entry["passage_loss"] = passage_loss(model, texts,
                        after=lambda: used_by_passage.append(installed.mixer.last_depths[0]))
                    calls = installed.take_calls()
                    installed.check_calls(calls, len(texts))
                    spread = installed.taken["passes_per_sequence"]
                    entry["passage_passes"] = {"per_passage": {str(k): v for k, v in spread.items()},
                        "mean": round(sum(k * v for k, v in spread.items()) / sum(spread.values()), 3),
                        "layer_calls": sum(calls)}
                # The prompts may all run to the budget, so passages that stopped early are recomputed by hand too.
                entry["early_stops"] = []
                early = [index for index, used in enumerate(used_by_passage) if used < budget]
                for index in early[:args.early]:
                    by_hand = explicit.logits(texts[index], design)
                    with install(model, design):
                        served = whole(texts[index])
                    row = {"passage": index, "passes_used": used_by_passage[index], "explicit_passes_used": explicit.passes_used,
                        "explicit": judged(compare([served], [by_hand]), label,
                            f"stopping disagrees with the explicit recomputation on passage {index}")}
                    if decided and explicit.passes_used != used_by_passage[index]:
                        failures.append(f"{label}: passage {index} stopped after {used_by_passage[index]} passes; "
                            f"the explicit recomputation after {explicit.passes_used}")
                    entry["early_stops"].append(row)
                entry["early_stops_found"] = len(early)
                for name, ids in prompts:
                    by_hand = explicit.logits(ids, design)
                    row = entry["prompts"][name]
                    with install(model, design):
                        served = whole(ids)
                    row["explicit"] = judged(compare([served], [by_hand]), label,
                        f"stopping disagrees with the explicit recomputation on {name}")
                    row["explicit_passes_used"] = explicit.passes_used
                    if decided and explicit.passes_used != row["passes_used"]:
                        failures.append(f"{label}: stopped after {row['passes_used']} passes on {name}; "
                            f"the explicit recomputation after {explicit.passes_used}")
                entry["calls_verified_every_forward"] = True
            except RuntimeError as exc:
                entry["error"] = str(exc)
                failures.append(f"{label}: {exc}")
            record["stopping"].append(entry)
            print(json.dumps(entry), flush=True)

        after = [bare(ids) for _, ids in prompts]
        same = all(bool((x == y).all()) for x, y in zip(before, after, strict=True))
        record["restored_after_removal"] = {"identical": same}
        if not same:
            failures.append("The model was not restored after the designs were removed")
        record.update(passed=not failures, failures=failures, completed_at=datetime.now(timezone.utc).isoformat())
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{profile}-{precision}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}; passed={not failures}", flush=True)
    for failure in failures:
        print("FAILED:", failure, flush=True)
    return not failures


def loss(args) -> None:
    """A plain parent's loss on the same passages, tokenized by its own tokenizer."""
    from recurrent_retrofit.loading import load_parent
    with gpu_lock(args.root):
        model, tokenizer, info = load_parent(args.profile, args.root, precision=args.precision)
        texts, text_facts = passages(args.root, tokenizer, args.passages)
        record = {"profile": args.profile, "precision": args.precision, "runtime": info, "passage_text": text_facts,
            "passage_loss": passage_loss(model, texts), "completed_at": datetime.now(timezone.utc).isoformat()}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"loss-{args.profile}-{args.precision}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profile": args.profile, **record["passage_loss"], "passages": text_facts["token_ids_sha256"][:12]}), flush=True)


def trace(args) -> None:
    """Pass by pass, by hand: the loss when the state is decoded there, how much of the state the gate
    replaced, and the head's confidence. Also the loss under two other readings of the gate, which
    shows whether the released arithmetic is the one the weights fit."""
    from recurrent_retrofit.loading import load_parent
    from recurrent_retrofit.reference import ExplicitGated
    import torch
    cross_entropy = torch.nn.functional.cross_entropy
    with gpu_lock(args.root):
        model, tokenizer, info = load_parent(args.profile, args.root, precision=args.precision)
        texts, text_facts = passages(args.root, tokenizer, args.passages)
        explicit = ExplicitGated(model)
        by_hand, layers = explicit.explicit, info["parent_layers"]
        start, end = info["gated"]["unit"]

        def unit(state):
            for index in range(start, end):
                state = by_hand.layer(index, state)
            return state

        def loss_of(state, ids) -> float:
            for index in range(end, layers):
                state = by_hand.layer(index, state)
            logits = model.lm_head(by_hand.core.norm(state))[0, :-1].float()
            return float(cross_entropy(logits, torch.tensor(ids[1:], device=logits.device), reduction="sum"))

        readings = ("as_released", "unit_output_without_the_gate", "gate_with_its_two_states_exchanged")
        losses = {name: [0.0] * args.budget for name in readings}
        replaced = [[0.0, 0.0] for _ in range(args.budget)]
        confidences: list[list[float]] = [[] for _ in range(args.budget)]
        scored = 0
        with torch.inference_mode():
            for ids in texts:
                entered = by_hand.prepare(ids)
                for index in range(start):
                    entered = by_hand.layer(index, entered)
                scored += len(ids) - 1
                state = entered
                for turn in range(args.budget):
                    output = unit(state)
                    keep = explicit.keep(output, state)
                    confidences[turn].append(explicit.confidence(output - state))
                    replaced[turn][0] += float(keep.float().mean())
                    replaced[turn][1] += float((keep > 0.99).float().mean())
                    losses["unit_output_without_the_gate"][turn] += loss_of(output, ids)
                    state = explicit.gate(output, state)
                    losses["as_released"][turn] += loss_of(state, ids)
                state = entered
                for turn in range(args.budget):
                    output = unit(state)
                    keep = explicit.keep(output, state)
                    state = (keep * state.to(keep.dtype) + (1 - keep) * output.to(keep.dtype)).to(output.dtype)
                    losses["gate_with_its_two_states_exchanged"][turn] += loss_of(state, ids)
        rows = []
        for turn in range(args.budget):
            values = sorted(confidences[turn])
            rows.append({"pass": turn + 1,
                "mean_loss": {name: round(losses[name][turn] / scored, 4) for name in readings},
                "mean_share_of_the_units_output_let_through": round(replaced[turn][0] / len(texts), 4),
                "share_of_entries_above_0.99": round(replaced[turn][1] / len(texts), 4),
                "confidence_at_the_last_token": {"lowest": round(values[0], 4), "median": round(values[len(values) // 2], 4),
                    "highest": round(values[-1], 4),
                    **{f"share_of_passages_above_{threshold}": round(sum(value > threshold for value in values) / len(values), 3)
                        for threshold in args.thresholds}}})
            print(json.dumps(rows[-1]), flush=True)
        record = {"profile": args.profile, "precision": args.precision, "runtime": info, "passage_text": text_facts,
            "what": "Each pass computed by hand on whole passages with no cache; the state after it is decoded",
            "passes": rows, "completed_at": datetime.now(timezone.utc).isoformat()}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"trace-{args.profile}-{args.precision}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}", flush=True)


def parents(args) -> None:
    """Which candidate parent the checkpoint's layers are closest to, tensor by tensor, and whether
    its tokenizer is the candidate's. Reads tensors one at a time on the CPU."""
    import os
    root = args.root.resolve()
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from recurrent_retrofit import loopus

    def snapshot(profile: str) -> Path:
        settings = profiles()[profile]
        return Path(snapshot_download(settings["repo_id"], revision=settings["revision"],
            cache_dir=str(root / "cache/huggingface/hub"), local_files_only=True))

    with gpu_lock(root):
        own = snapshot(args.profile)
        config = loopus.read_config(own)
        tokenizer = loopus.tokenizer_of(own)
        record = {"profile": args.profile, "checkpoint": identity(args.profile),
            "parent_named_in_config": config.get("base_model_name_or_path"),
            "parent_named_in_model_card": next((line.strip("- ").strip() for line in
                (own / "README.md").read_text(encoding="utf-8").split("base_model:", 1)[-1].splitlines() if line.strip()), None),
            "candidates": {}}
        samples = ["The capital of France is Paris.\nQuestion: 2+2?", " leading space", "naïve café — 日本語",
            "<|im_start|>user\nhi<|im_end|>\n", "<think>\n\n</think>\n\n", "A\tB\r\nC  D", "<|endoftext|>"]
        with safe_open(str(own / "model.safetensors"), "pt") as mine:
            keys = [key for key in mine.keys() if loopus.rename(key, config)[0] == "model"]
            for candidate in args.candidates:
                folder = snapshot(candidate)
                handles = [safe_open(str(path), "pt") for path in sorted(folder.glob("*.safetensors"))]
                where = {key: handle for handle in handles for key in handle.keys()}
                blocks: dict[str, list[float]] = {}
                identical = 0
                for key in keys:
                    target = loopus.rename(key, config)[1]
                    # A parent with tied weights stores its output head as the embedding.
                    source = target if target in where else "model.embed_tokens.weight"
                    theirs = where[source].get_tensor(source).float()
                    ours = mine.get_tensor(key).float()
                    block = "output head" if target == "lm_head.weight" else "embedding" if "embed" in target \
                        else "final norm" if target == "model.norm.weight" else key.split(".")[0] + " layers"
                    sums = blocks.setdefault(block, [0.0, 0.0, 0, 0])
                    sums[0] += float((ours - theirs).pow(2).sum())
                    sums[1] += float(theirs.pow(2).sum())
                    sums[2] += 1
                    same = bool((ours == theirs).all())
                    sums[3] += same
                    identical += same
                other = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
                record["candidates"][candidate] = {
                    "model": identity(candidate), "tensors_compared": len(keys), "tensors_identical": identical,
                    "relative_distance": round(math.sqrt(sum(s[0] for s in blocks.values()) / sum(s[1] for s in blocks.values())), 6),
                    "by_block": {block: {"tensors": s[2], "identical": s[3], "relative_distance": round(math.sqrt(s[0] / s[1]), 6)}
                        for block, s in blocks.items()},
                    "tokenizer": {"same_vocabulary": tokenizer.get_vocab() == other.get_vocab(),
                        "same_chat_template": tokenizer.chat_template == other.chat_template,
                        "same_end_and_padding_tokens": (tokenizer.eos_token_id, tokenizer.pad_token_id)
                            == (other.eos_token_id, other.pad_token_id),
                        "sample_texts_encoded_identically": sum(tokenizer.encode(text) == other.encode(text) for text in samples),
                        "sample_texts": len(samples)}}
                print(json.dumps({candidate: record["candidates"][candidate]}), flush=True)
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"parents-{args.profile}.json"
    destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {destination}", flush=True)


def compact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.name.startswith("parents-"):
        return {"record": path.name, "record_sha256": digest, "kind": "parents", **data}
    runtime = data["runtime"]
    shared = {"record": path.name, "record_sha256": digest, "model": runtime["parent"], "precision": data["precision"],
        "versions": runtime["versions"], "gpu": runtime["gpu"], "completed_at": data["completed_at"],
        "passage_text": data["passage_text"]}
    if path.name.startswith("loss-"):
        return {**shared, "kind": "parent loss", "profile": data["profile"], "passage_loss": data["passage_loss"]}
    if path.name.startswith("trace-"):
        return {**shared, "kind": "pass by pass", "profile": data["profile"], "what": data["what"], "passes": data["passes"]}

    def speed(rows: dict) -> float:
        return round(sum(row["decode_tokens_per_second"] for row in rows.values()) / len(rows), 2)

    def peak(rows: dict):
        peaks = [row["peak_allocated_GiB"] for row in rows.values() if row["peak_allocated_GiB"]]
        return round(max(peaks), 2) if peaks else None

    def fixed(entry: dict) -> dict:
        result = {"design": entry["design"], "passes": entry["passes"], "error": entry.get("error"),
            "layer_calls_per_forward": entry["layer_calls_per_forward"],
            "calls_and_cache_verified_every_forward": entry.get("calls_and_cache_verified_every_forward", False)}
        if "error" not in entry:
            result.update({
                "cache_gap": entry["cache_control"]["max_absolute_logit_difference"],
                "cache_argmax_equal": f"{entry['cache_control']['argmax_equal']}/{entry['cache_control']['steps']}",
                "explicit_gap": max(entry["explicit"]["max_absolute_logit_difference"],
                    entry["explicit_every_position"]["max_absolute_logit_difference"]),
                "explicit_argmax_equal": f"{entry['explicit']['argmax_equal']}/{entry['explicit']['steps']}",
                "passage_loss": entry["passage_loss"],
                "mean_decode_tokens_per_second": speed(entry["rows"]), "peak_allocated_GiB": peak(entry["rows"]),
                "texts": {label: row["text"] for label, row in entry["rows"].items()}})
        return result

    def stopping(entry: dict) -> dict:
        result = {"design": entry["design"], "passes": entry["passes"], "threshold": entry["threshold"],
            "error": entry.get("error"), "calls_verified_every_forward": entry.get("calls_verified_every_forward", False)}
        if "error" not in entry:
            rows = entry["prompts"]
            result.update({
                "passes_used": {name: row["passes_used"] for name, row in rows.items()},
                "explicit_passes_used": {name: row["explicit_passes_used"] for name, row in rows.items()},
                "explicit_gap": max(row["explicit"]["max_absolute_logit_difference"] for row in rows.values()),
                "identical_to_the_fixed_loop": all(row["against_the_fixed_loop"]["identical"] for row in rows.values())
                    if entry["threshold"] == 1.0 else None,
                "identical_to_one_pass": all(row["against_one_pass"]["identical"] for row in rows.values())
                    if entry["threshold"] == 0.0 else None,
                "passage_loss": entry["passage_loss"], "passage_passes": entry["passage_passes"],
                "early_stops_found": entry["early_stops_found"],
                "early_stops_recomputed_by_hand": [{"passage": row["passage"], "passes_used": row["passes_used"],
                    "explicit_passes_used": row["explicit_passes_used"],
                    "explicit_gap": row["explicit"]["max_absolute_logit_difference"]} for row in entry["early_stops"]]})
        return result

    return {**shared, "kind": "gated loop", "profile": data["profile"], "placement": data["placement"],
        "passed": data["passed"], "failures": data["failures"], "checkpoint": runtime["gated"],
        "parent_layers": runtime["parent_layers"], "tokens_per_prompt": data["tokens_per_prompt"],
        "prompts": data["prompts"], "gaps_decided": data["precision"] == "float32",
        "tolerance": data["tolerances"]["fp32_max_absolute_logit_difference"],
        "restored_after_removal": data["restored_after_removal"]["identical"],
        "fixed": [fixed(entry) for entry in data["fixed"]], "stopping": [stopping(entry) for entry in data["stopping"]]}


def summarize(args) -> None:
    records = sorted(args.input.resolve().glob("*.json"))
    if not records:
        raise SystemExit("No acceptance records found")
    entries = [compact(path) for path in records]
    summary = {"schema_version": 1,
        "purpose": "Acceptance of a checkpoint trained to loop; validates the tool, not the method",
        "source_directory": args.input.as_posix(), "note": args.note, "records": entries}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checked = [entry for entry in entries if entry["kind"] == "gated loop"]
    print(f"Summarized {len(entries)} records; all checks passed: {all(entry['passed'] for entry in checked)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    looped = [name for name in profiles() if needs_gate(name)]
    plain = [name for name, settings in profiles().items() if settings.get("family") == "qwen3"]
    root = {"type": Path, "default": Path.home() / "llm/recurrent"}
    output = {"type": Path, "default": ROOT / "artifacts/runs/local-runtime"
        / datetime.now(timezone.utc).strftime("gated-check-%Y%m%dT%H%M%SZ")}
    command = commands.add_parser("parents", help="Compare the checkpoint with candidate parents")
    command.add_argument("--profile", required=True, choices=looped)
    command.add_argument("--candidates", nargs="+", required=True, choices=plain)
    command.add_argument("--root", **root)
    command.add_argument("--output", **output)
    command.set_defaults(handler=parents)
    command = commands.add_parser("run", help="Check the checkpoint's gated loop")
    command.add_argument("--profile", required=True, choices=looped)
    command.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    command.add_argument("--placement", choices=["gpu", "cpu"], default="gpu")
    command.add_argument("--tokens", type=int, default=24)
    command.add_argument("--passes", type=int, nargs="+", default=[1, 2, 3, 4, 8, 20], help="Fixed numbers of passes to check")
    command.add_argument("--budget", type=int, default=8, help="The most passes when stopping")
    command.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.9], help="Stopping thresholds to check")
    command.add_argument("--passages", type=int, default=200)
    command.add_argument("--early", type=int, default=5, help="How many early-stopping passages to recompute by hand")
    command.add_argument("--root", **root)
    command.add_argument("--output", **output)
    command.set_defaults(handler=lambda args: sys.exit(0 if check(args) else 1))
    command = commands.add_parser("trace", help="What each pass does, computed by hand")
    command.add_argument("--profile", required=True, choices=looped)
    command.add_argument("--precision", choices=["bfloat16", "float32"], default="float32")
    command.add_argument("--budget", type=int, default=8, help="How many passes to trace")
    command.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.9])
    command.add_argument("--passages", type=int, default=200)
    command.add_argument("--root", **root)
    command.add_argument("--output", **output)
    command.set_defaults(handler=trace)
    command = commands.add_parser("loss", help="A plain parent's loss on the same passages")
    command.add_argument("--profile", required=True, choices=plain)
    command.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    command.add_argument("--passages", type=int, default=200)
    command.add_argument("--root", **root)
    command.add_argument("--output", **output)
    command.set_defaults(handler=loss)
    command = commands.add_parser("summarize", help="Write the compact summary of a directory of records")
    command.add_argument("input", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--note", help="A caveat to store with the summary")
    command.set_defaults(handler=summarize)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
