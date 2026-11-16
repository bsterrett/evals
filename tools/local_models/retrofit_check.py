"""Acceptance checks for the retrofit runner on a real parent checkpoint (WSL).

For one parent profile this checks that:

1. the parent program, and one pass of any design, reproduce the parent's logits exactly;
2. every forward of every design executes the program's layer calls and fills
   one cache slot per position, in prefill and in decoding;
3. a plain loop is identical to a physically deeper stock model, with and
   without a cache;
4. cached decoding agrees with recomputing the whole prefix without a cache;
5. removing the program restores the parent exactly;
6. every design, including each update rule, agrees with an explicit
   recomputation that calls the parent's layers by hand.

Checks 1, 2, 3 and 5 are exact in any precision. Checks 4 and 6 compare
different orders of the same arithmetic, so they are decided in FP32 only: BF16
rounding is amplified by some loops (layer-local loops especially) to gaps of
several logits, which a threshold cannot tell from an error. In BF16 the gaps
are recorded beside the parent's own, and not judged. Run FP32 on the GPU when
twice the weights fit, otherwise with `--placement cpu`.

Plain loops use the middle third of the layers and the update rules the middle
four. These are mechanical choices for the checks, not tuned or recommended
placements, and the generated text is saved only to show that the models run.
This validates the tool. It is not evidence that any design helps or harms a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import LocalRequest, encode_prompt, profiles
from eval_core.local.worker import gpu_lock
from recurrent_retrofit.program import Update, block_loop, layer_loop, parent

PROMPTS = [
    ("raw-capital", "raw-v1", "The capital of France is"),
    ("chat-sum", "native-chat-no-thinking-v1", "What is 2 + 2? Reply with just the number."),
    ("chat-colors", "native-chat-no-thinking-v1", "Name the three primary colors in one short sentence."),
]
FP32_TOLERANCE = 0.01
CONTROL_STEPS = 6
UPDATES = [
    (block_loop, Update("euler")),
    (block_loop, Update("euler", anchor=0.5)),
    (layer_loop, Update("euler", anchor=0.5)),
    (block_loop, Update("uniform")),
    (block_loop, Update("uniform", per="layer")),
    (layer_loop, Update("uniform")),
    (block_loop, Update("ema", eta=0.5)),
]


def compare(a: dict, b: dict, steps: int | None = None) -> dict:
    pairs = list(zip(a["logits"], b["logits"]))[:steps]
    return {"steps": len(pairs),
        "identical": all(bool((x == y).all()) for x, y in pairs),
        "argmax_equal": sum(int(x.argmax()) == int(y.argmax()) for x, y in pairs),
        "max_absolute_logit_difference": max(float((x - y).abs().max()) for x, y in pairs)}


def check(profile: str, unit: tuple[int, int] | None, passes: list[int], update_passes: list[int], tokens: int,
        prompt_count: int, root: Path, output: Path, precision: str, placement: str) -> bool:
    from recurrent_retrofit.loading import load_parent
    from recurrent_retrofit.reference import Explicit, StockCopy
    from recurrent_retrofit.runner import Retrofit, greedy
    import torch
    failures: list[str] = []
    on_gpu = placement == "gpu"
    decided = precision == "float32"
    with gpu_lock(root):
        model, tokenizer, info = load_parent(profile, root, precision=precision, placement=placement)
        torch.use_deterministic_algorithms(True)
        layers = info["parent_layers"]
        start, end = unit or (layers // 3, 2 * layers // 3)
        small = (layers // 2 - 2, layers // 2 + 2)
        prompts = [(label, encode_prompt(tokenizer, LocalRequest(info["parent"], 1, text, tokens, 0, fmt)))
            for label, fmt, text in PROMPTS[:prompt_count]]
        first_label, first_ids = prompts[0]
        steps = min(tokens, CONTROL_STEPS)
        explicit = Explicit(model)
        decode = lambda out: tokenizer.decode(out["tokens"], skip_special_tokens=False)

        def measured(retrofit: Retrofit | None) -> dict:
            """Cached greedy decoding of every prompt, instrumented when a program is installed."""
            rows, outputs = {}, {}
            for label, ids in prompts:
                stamps = []
                def after(step, cache):
                    if retrofit is not None:
                        calls = retrofit.take_calls()
                        retrofit.check_calls(calls)
                        retrofit.check_cache(cache, len(ids) + step)
                    if on_gpu:
                        torch.cuda.synchronize()
                    stamps.append(time.perf_counter())
                if retrofit is not None:
                    retrofit.take_calls()
                if on_gpu:
                    torch.cuda.reset_peak_memory_stats()
                out = greedy(model, ids, tokens, after_forward=after)
                outputs[label] = out
                rows[label] = {"input_tokens": len(ids), "output_token_ids": out["tokens"], "text": decode(out),
                    "decode_tokens_per_second": (len(stamps) - 1) / (stamps[-1] - stamps[0]),
                    "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3 if on_gpu else None}
            return {"rows": rows, "outputs": outputs}

        def judged(result: dict, label: str, what: str) -> dict:
            result["decided"] = decided
            if decided and result["max_absolute_logit_difference"] > FP32_TOLERANCE:
                failures.append(f"{label}: {what}")
            return result

        def same_as_parent(program) -> bool:
            with Retrofit(model, program) as retrofit:
                got = measured(retrofit)
            return all(compare(got["outputs"][label], baseline["outputs"][label])["identical"] for label, _ in prompts)

        def examine(program, exact_reference: bool) -> dict:
            label = f"{program.spec} x{program.passes}"
            entry = {"program": program.spec, "passes": program.passes, "layer_calls": program.layer_calls,
                "update_cells": len(program.cells())}
            try:
                with Retrofit(model, program) as retrofit:
                    got = measured(retrofit)
                    cached = got["outputs"][first_label]
                    full = greedy(model, first_ids, steps, cached=False, forced=cached["tokens"])
                entry["rows"] = got["rows"]
                entry["calls_and_slots_verified_every_forward"] = True
                entry["differs_from_parent"] = compare(cached, baseline["outputs"][first_label], 1)
                entry["cache_control"] = judged(compare(cached, full, steps), label,
                    "cached decoding disagrees with the full-prefix control")
                by_hand = {"logits": [explicit.logits(first_ids, program)]}
                entry["explicit"] = judged(compare(full, by_hand, 1), label,
                    "the runner disagrees with the explicit recomputation")
                if exact_reference:
                    with StockCopy(model, program):
                        entry["stock_copy_cached"] = compare(cached,
                            greedy(model, first_ids, steps, forced=cached["tokens"]), steps)
                        entry["stock_copy_uncached"] = compare(full,
                            greedy(model, first_ids, steps, cached=False, forced=cached["tokens"]), steps)
                    if not (entry["stock_copy_cached"]["identical"] and entry["stock_copy_uncached"]["identical"]):
                        failures.append(f"{label}: runner differs from the deeper stock model")
            except RuntimeError as exc:
                entry["error"] = str(exc)
                failures.append(f"{label}: {exc}")
            print(json.dumps({k: v for k, v in entry.items() if k != "rows"}), flush=True)
            for name, row in entry.get("rows", {}).items():
                print(f"    {name}: {row['text']!r}", flush=True)
            return entry

        measured(None)  # warm-up, discarded
        baseline = measured(None)
        record = {"profile": profile, "precision": precision, "placement": placement, "unit": [start, end],
            "update_unit": list(small), "tokens_per_prompt": tokens, "prompts": [label for label, _ in prompts],
            "runtime": info, "tolerances": {"fp32_max_absolute_logit_difference": FP32_TOLERANCE},
            "parent": baseline["rows"], "parent_equivalents": [], "loops": [], "updates": []}
        own = baseline["outputs"][first_label]
        record["parent_cache_control"] = judged(compare(own,
            greedy(model, first_ids, steps, cached=False, forced=own["tokens"]), steps), "parent",
            "cached decoding disagrees with the full-prefix control")
        record["parent_explicit"] = judged(compare(
            greedy(model, first_ids, 1, cached=False), {"logits": [explicit.logits(first_ids, parent(layers))]}, 1),
            "parent", "the explicit recomputation disagrees with the parent")

        one_pass = [parent(layers), block_loop(layers, start, end, 1), layer_loop(layers, start, end, 1)]
        one_pass += [build(layers, *small, 1, update) for build, update in UPDATES]
        for program in one_pass:
            same = same_as_parent(program)
            record["parent_equivalents"].append({"program": program.spec, "passes": 1, "identical": same})
            if not same:
                failures.append(f"{program.spec} with one pass differs from the parent")
            print(f"{profile} {precision}: {program.spec} x1 identical to parent: {same}", flush=True)

        for build in (block_loop, layer_loop):
            for count in passes:
                record["loops"].append(examine(build(layers, start, end, count), exact_reference=True))
        for build, update in UPDATES:
            for count in update_passes:
                record["updates"].append(examine(build(layers, *small, count, update), exact_reference=False))

        restored = measured(None)
        same = all(compare(restored["outputs"][label], baseline["outputs"][label])["identical"] for label, _ in prompts)
        record["restored_parent"] = {"identical": same}
        if not same:
            failures.append("The parent was not restored after the programs were removed")
        record.update(passed=not failures, failures=failures, completed_at=datetime.now(timezone.utc).isoformat())
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{profile}-{precision}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}; passed={not failures}", flush=True)
    for failure in failures:
        print("FAILED:", failure, flush=True)
    return not failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True,
        choices=[name for name, settings in profiles().items() if settings.get("family") == "qwen3"])
    parser.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--placement", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--unit", help="start:end of the layers plain loops repeat; default is the middle third")
    parser.add_argument("--passes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--update-passes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--prompts", type=int, default=len(PROMPTS), choices=range(1, len(PROMPTS) + 1),
        help="How many of the prompts to decode; the controls always use the first")
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    unit = tuple(int(value) for value in args.unit.split(":")) if args.unit else None
    output = args.output or ROOT / "artifacts/runs/local-runtime" / datetime.now(timezone.utc).strftime("retrofit-check-%Y%m%dT%H%M%SZ")
    passed = check(args.profile, unit, args.passes, args.update_passes, args.tokens, args.prompts, args.root,
        output, args.precision, args.placement)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
