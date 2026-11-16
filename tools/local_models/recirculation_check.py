"""Acceptance checks for recirculation on a real parent checkpoint (WSL).

  run        check one parent profile and save the full record
  summarize  write the compact, tracked summary of a directory of records

For one parent profile `run` checks that:

1. one pass of a recirculation design is identical to the parent decoded one
   token at a time, prompt included;
2. two passes with alpha 0 are identical to one pass;
3. every forward pass makes the design's layer calls for each of its tokens and
   leaves one cache entry per token in every layer;
4. cached decoding agrees with recomputing the whole prefix without a cache;
5. every design agrees with an explicit recomputation that calls the parent's
   layers by hand, with no cache and none of the runner's code;
6. removing the design restores the parent exactly.

Checks 1, 2, 3 and 6 are exact in any precision. Checks 4 and 5 compare
different orders of the same arithmetic, so they are decided in FP32 only; in
BF16 the gaps are recorded beside the parent's own and not judged. Run FP32 on
the GPU when twice the weights fit, otherwise with `--placement cpu`.

The designs put the source at half depth and the destination at a quarter, plus
one that feeds the last layer back to the first. These are mechanical choices
for the checks, not tuned or recommended placements, and the generated text is
saved only to show that the models run. This validates the tool. It is not
evidence that any design helps or harms a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import LocalRequest, encode_prompt, profiles
from eval_core.local.worker import gpu_lock
from recurrent_retrofit.program import Recirculation

PROMPTS = [
    ("raw-capital", "raw-v1", "The capital of France is"),
    ("chat-sum", "native-chat-no-thinking-v1", "What is 2 + 2? Reply with just the number."),
    ("chat-colors", "native-chat-no-thinking-v1", "Name the three primary colors in one short sentence."),
]
FP32_TOLERANCE = 0.01
CONTROL_STEPS = 6


def designs(layers: int) -> list[Recirculation]:
    source, destination = layers // 2, layers // 4
    return [
        Recirculation(layers, source, destination, 0.07),
        Recirculation(layers, source, destination, 0.07, "additive"),
        Recirculation(layers, source, destination, 0.15),
        Recirculation(layers, layers - 1, 0, 0.07),
    ]


def compare(a: list, b: list) -> dict:
    pairs = list(zip(a, b, strict=True))
    return {"steps": len(pairs),
        "identical": all(bool((x == y).all()) for x, y in pairs),
        "argmax_equal": sum(int(x.argmax()) == int(y.argmax()) for x, y in pairs),
        "max_absolute_logit_difference": max(float((x - y).abs().max()) for x, y in pairs)}


def check(profile: str, tokens: int, prompt_count: int, root: Path, output: Path, precision: str, placement: str) -> bool:
    from recurrent_retrofit.loading import load_parent
    from recurrent_retrofit.reference import ExplicitRecirculation
    from recurrent_retrofit.runner import greedy, install
    import torch
    from transformers import DynamicCache
    failures: list[str] = []
    on_gpu = placement == "gpu"
    decided = precision == "float32"
    with gpu_lock(root):
        model, tokenizer, info = load_parent(profile, root, precision=precision, placement=placement)
        torch.use_deterministic_algorithms(True)
        layers = info["parent_layers"]
        device = next(model.parameters()).device
        prompts = [(label, encode_prompt(tokenizer, LocalRequest(info["parent"], 1, text, tokens, 0, fmt)))
            for label, fmt, text in PROMPTS[:prompt_count]]
        first_label, first_ids = prompts[0]
        steps = min(tokens, CONTROL_STEPS)
        explicit = ExplicitRecirculation(model)
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)

        def token_by_token(ids: list[int], forced: list[int]) -> list:
            """The bare parent, given the prompt and then `forced` one token at a time."""
            cache, logits = DynamicCache(config=model.config), []
            with torch.inference_mode():
                for index, token in enumerate(ids + forced[:-1]):
                    out = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
                    if index >= len(ids) - 1:
                        logits.append(out.logits[0, -1].float().cpu())
            return logits

        def measured(installed) -> dict:
            """Cached greedy decoding of every prompt, instrumented when a design is installed."""
            rows, outputs = {}, {}
            for label, ids in prompts:
                stamps = []
                def after(step, cache):
                    if installed is not None:
                        calls = installed.take_calls()
                        installed.check_calls(calls, installed.runs_for(len(ids) if step == 0 else 1))
                        installed.check_cache(cache, len(ids) + step)
                    if on_gpu:
                        torch.cuda.synchronize()
                    stamps.append(time.perf_counter())
                if installed is not None:
                    installed.take_calls()
                if on_gpu:
                    torch.cuda.reset_peak_memory_stats()
                out = greedy(model, ids, tokens, after_forward=after)
                outputs[label] = out
                rows[label] = {"input_tokens": len(ids), "output_token_ids": out["tokens"], "text": decode(out["tokens"]),
                    "decode_tokens_per_second": (len(stamps) - 1) / (stamps[-1] - stamps[0]),
                    "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3 if on_gpu else None}
            return {"rows": rows, "outputs": outputs}

        def judged(result: dict, label: str, what: str) -> dict:
            result["decided"] = decided
            if decided and result["max_absolute_logit_difference"] > FP32_TOLERANCE:
                failures.append(f"{label}: {what}")
            return result

        def exact(result: dict, label: str, what: str) -> dict:
            if not result["identical"]:
                failures.append(f"{label}: {what}")
            return result

        measured(None)  # warm-up, discarded
        baseline = measured(None)
        own = baseline["outputs"][first_label]
        record = {"profile": profile, "precision": precision, "placement": placement, "tokens_per_prompt": tokens,
            "prompts": [label for label, _ in prompts], "runtime": info,
            "tolerances": {"fp32_max_absolute_logit_difference": FP32_TOLERANCE},
            "parent": baseline["rows"], "designs": []}
        # How far the parent's own two orders of arithmetic are apart, for scale.
        record["parent_cache_control"] = judged(compare(own["logits"][:steps],
            greedy(model, first_ids, steps, cached=False, forced=own["tokens"])["logits"]), "parent",
            "cached decoding disagrees with the full-prefix control")
        record["parent_token_by_token_against_parallel_prefill"] = judged(compare(
            token_by_token(first_ids, own["tokens"][:steps]), own["logits"][:steps]), "parent",
            "token-by-token decoding disagrees with parallel prefill")

        one = Recirculation(layers, layers // 2, layers // 4, 0.07, passes=1)
        with install(model, one) as installed:
            once = measured(installed)
        record["one_pass"] = {"design": one.spec, "rows": once["rows"], "calls_and_cache_verified_every_forward": True,
            "against_parent_token_by_token": {label: exact(compare(
                once["outputs"][label]["logits"], token_by_token(ids, once["outputs"][label]["tokens"])),
                f"{one.spec} x1 on {label}", "one pass differs from the parent token by token")
                for label, ids in prompts}}
        print(json.dumps({"one_pass": record["one_pass"]["against_parent_token_by_token"]}), flush=True)

        zero = Recirculation(layers, layers // 2, layers // 4, 0.0)
        with install(model, zero) as installed:
            nothing = measured(installed)
        record["zero_alpha"] = {"design": zero.spec, "against_one_pass": {label: exact(compare(
            nothing["outputs"][label]["logits"], once["outputs"][label]["logits"]),
            f"{zero.spec} x2 on {label}", "alpha 0 differs from one pass") for label, _ in prompts}}
        print(json.dumps({"zero_alpha": record["zero_alpha"]["against_one_pass"]}), flush=True)

        for design in designs(layers):
            label = f"{design.spec} x{design.passes}"
            entry = {"design": design.spec, "passes": design.passes, "layer_calls_per_token": design.layer_calls}
            try:
                with install(model, design) as installed:
                    got = measured(installed)
                    cached = got["outputs"][first_label]
                    full = greedy(model, first_ids, steps, cached=False, forced=cached["tokens"])
                entry["rows"] = got["rows"]
                entry["calls_and_cache_verified_every_forward"] = True
                entry["differs_from_one_pass"] = compare(cached["logits"][:1], once["outputs"][first_label]["logits"][:1])
                entry["cache_control"] = judged(compare(cached["logits"][:steps], full["logits"]), label,
                    "cached decoding disagrees with the full-prefix control")
                by_hand = explicit.logits(first_ids + cached["tokens"][:steps - 1], design)[len(first_ids) - 1:]
                entry["explicit"] = judged(compare(cached["logits"][:steps], by_hand), label,
                    "the runner disagrees with the explicit recomputation")
            except RuntimeError as exc:
                entry["error"] = str(exc)
                failures.append(f"{label}: {exc}")
            record["designs"].append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k != "rows"}), flush=True)
            for name, row in entry.get("rows", {}).items():
                print(f"    {name}: {row['text']!r}", flush=True)

        restored = measured(None)
        same = all(compare(restored["outputs"][label]["logits"], baseline["outputs"][label]["logits"])["identical"]
            for label, _ in prompts)
        record["restored_parent"] = {"identical": same}
        if not same:
            failures.append("The parent was not restored after the designs were removed")
        record.update(passed=not failures, failures=failures, completed_at=datetime.now(timezone.utc).isoformat())
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{profile}-{precision}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}; passed={not failures}", flush=True)
    for failure in failures:
        print("FAILED:", failure, flush=True)
    return not failures


def compact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    runtime = data["runtime"]

    def speed(rows: dict) -> float:
        return round(sum(row["decode_tokens_per_second"] for row in rows.values()) / len(rows), 2)

    def peak(rows: dict):
        peaks = [row["peak_allocated_GiB"] for row in rows.values() if row["peak_allocated_GiB"]]
        return round(max(peaks), 2) if peaks else None

    def design(entry: dict) -> dict:
        result = {"design": entry["design"], "passes": entry["passes"],
            "layer_calls_per_token": entry["layer_calls_per_token"], "error": entry.get("error"),
            "calls_and_cache_verified_every_forward": entry.get("calls_and_cache_verified_every_forward", False)}
        if "error" not in entry:
            result.update({
                "cache_gap": entry["cache_control"]["max_absolute_logit_difference"],
                "cache_argmax_equal": f"{entry['cache_control']['argmax_equal']}/{entry['cache_control']['steps']}",
                "explicit_gap": entry["explicit"]["max_absolute_logit_difference"],
                "explicit_argmax_equal": f"{entry['explicit']['argmax_equal']}/{entry['explicit']['steps']}",
                "first_step_gap_from_one_pass": entry["differs_from_one_pass"]["max_absolute_logit_difference"],
                "mean_decode_tokens_per_second": speed(entry["rows"]), "peak_allocated_GiB": peak(entry["rows"]),
                "texts": {label: row["text"] for label, row in entry["rows"].items()}})
        return result

    return {
        "record": path.name, "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "passed": data["passed"], "failures": data["failures"], "completed_at": data["completed_at"],
        "parent": runtime["parent"], "parent_layers": runtime["parent_layers"],
        "precision": data["precision"], "placement": data["placement"],
        "tokens_per_prompt": data["tokens_per_prompt"], "prompts": data["prompts"],
        "versions": runtime["versions"], "gpu": runtime["gpu"],
        "gaps_decided": data["precision"] == "float32",
        "tolerance": data["tolerances"]["fp32_max_absolute_logit_difference"],
        "parent_cache_gap": data["parent_cache_control"]["max_absolute_logit_difference"],
        "parent_token_by_token_gap_from_parallel_prefill":
            data["parent_token_by_token_against_parallel_prefill"]["max_absolute_logit_difference"],
        "parent_mean_decode_tokens_per_second": speed(data["parent"]), "parent_peak_allocated_GiB": peak(data["parent"]),
        "parent_texts": {label: row["text"] for label, row in data["parent"].items()},
        "one_pass_design": data["one_pass"]["design"],
        "one_pass_identical_to_parent_token_by_token":
            {label: result["identical"] for label, result in data["one_pass"]["against_parent_token_by_token"].items()},
        "one_pass_mean_decode_tokens_per_second": speed(data["one_pass"]["rows"]),
        "zero_alpha_identical_to_one_pass":
            {label: result["identical"] for label, result in data["zero_alpha"]["against_one_pass"].items()},
        "parent_restored_after_removal": data["restored_parent"]["identical"],
        "designs": [design(entry) for entry in data["designs"]],
    }


def summarize(args) -> None:
    records = sorted(args.input.resolve().glob("*.json"))
    if not records:
        raise SystemExit("No acceptance records found")
    summary = {"schema_version": 1,
        "purpose": "Recirculation acceptance; validates the tool, not whether any design helps a model",
        "source_directory": args.input.as_posix(), "note": args.note,
        "records": [compact(path) for path in records]}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summarized {len(records)} records; all passed: {all(r['passed'] for r in summary['records'])}")


def run(args) -> None:
    output = args.output or ROOT / "artifacts/runs/local-runtime" / datetime.now(timezone.utc).strftime("recirculation-check-%Y%m%dT%H%M%SZ")
    sys.exit(0 if check(args.profile, args.tokens, args.prompts, args.root, output, args.precision, args.placement) else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="Check one parent profile")
    command.add_argument("--profile", required=True,
        choices=[name for name, settings in profiles().items() if settings.get("family") == "qwen3"])
    command.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    command.add_argument("--placement", choices=["gpu", "cpu"], default="gpu")
    command.add_argument("--tokens", type=int, default=24)
    command.add_argument("--prompts", type=int, default=len(PROMPTS), choices=range(1, len(PROMPTS) + 1),
        help="How many of the prompts to decode; the controls always use the first")
    command.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    command.add_argument("--output", type=Path)
    command.set_defaults(handler=run)
    command = commands.add_parser("summarize", help="Write the compact summary of a directory of records")
    command.add_argument("input", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--note", help="A caveat to store with the summary")
    command.set_defaults(handler=summarize)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
