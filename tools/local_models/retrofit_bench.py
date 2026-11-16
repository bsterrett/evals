"""Benchmark a retrofit design or its parent with lm-eval, and pair two runs.

  fetch    download the pinned benchmark datasets; the only step that uses the network
  run      run benchmark tasks on a parent, or on a design installed on it
  compare  count items by outcome for two runs of the same tasks
  summarize  write the compact, tracked summary of a directory of run records

A run keeps every item's result, so two runs can be paired item by item. The
comparison reports counts and scores. It applies no rule for calling a
difference an improvement.

Run inside the WSL `retrofit` environment. Examples:

  retrofit_bench.py fetch
  retrofit_bench.py run --profile qwen3-4b-instruct --tasks commonsense_qa --fewshot 7 --output parent.json
  retrofit_bench.py run --profile qwen3-4b-instruct --method 'block[15:19]/euler' --passes 3 \
      --tasks commonsense_qa --fewshot 7 --output design.json
  retrofit_bench.py compare parent.json design.json

A checkpoint trained to loop always takes its gated design, and is compared with
the parent it was trained from, which is another checkpoint:

  retrofit_bench.py run --profile qwen3-1.7b --tasks openbookqa --output parent.json
  retrofit_bench.py run --profile qwen3-1.7b-loopus --method 'gated[2:27]/halt,threshold=0.6' --passes 8 \
      --tasks openbookqa --output looped.json
  retrofit_bench.py compare parent.json looped.json --across-checkpoints
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import profiles
from recurrent_retrofit import bench
from recurrent_retrofit.backend import retrofit_identity
from recurrent_retrofit.loading import FAMILIES
from recurrent_retrofit.program import from_spec

SAME = ("settings", "datasets", "harness")


def fetch(args) -> None:
    rows = bench.fetch(args.root, args.tasks or sorted(bench.PINS["tasks"]))
    for task, splits in rows.items():
        print(f"{task}: {bench.pin(task)['dataset']}@{bench.pin(task)['revision'][:12]} {splits}", flush=True)


def run(args) -> None:
    root = args.root.resolve()
    if args.passes > 1 and not args.method:
        raise SystemExit("More than one pass needs --method")
    datasets = bench.check_pins(root, args.tasks)
    from recurrent_retrofit.loading import load_parent, needs_gate
    # Reject a malformed or misplaced design before the model loads.
    draft = from_spec(args.method, 10**6, args.passes) if args.method else None
    if needs_gate(args.profile) != (draft is not None and draft.kind == "gated"):
        raise SystemExit("A checkpoint trained to loop runs with its gated design, e.g. --method 'gated[2:27]', "
            "and only it does")
    if getattr(draft, "threshold", None) is not None and args.batch_size != 1:
        raise SystemExit("With a stopping threshold use batch size 1: the harness pads rows without a mask, "
            "and a padded row would be stopped by its padding")
    import torch
    from eval_core.local.worker import gpu_lock
    from recurrent_retrofit.runner import install
    with gpu_lock(root):
        model, tokenizer, info = load_parent(args.profile, root, precision=args.precision)
        program = from_spec(args.method, info["parent_layers"], args.passes) if args.method else None
        retrofit = install(model, program) if program else None
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        results = bench.evaluate(model, tokenizer, args.tasks, fewshot=args.fewshot, limit=args.limit,
            batch_size=args.batch_size, chat_template=args.chat_template, seed=args.seed,
            max_length=args.max_length, rolling_context=args.rolling_context)
        seconds = time.perf_counter() - start
        executed = None
        if retrofit:
            # The hooks on the real parent layers counted every call during the run.
            # A recirculation runs once per token, so its "forward passes" are token steps.
            calls = retrofit.take_calls()
            forwards = retrofit.forwards_in(calls)
            retrofit.check_calls(calls, forwards)
            executed = {"forward_passes": forwards, "layer_calls_per_forward": program.layer_calls,
                "a_forward_pass_is": "one token" if program.kind == "recirculate" else "one call of the model",
                "every_layer_matched_the_program": True}
            if getattr(program, "threshold", None) is not None:
                # With stopping the count per forward pass varies; the gate's own tally was checked against the hooks.
                used = retrofit.taken["passes_per_sequence"]
                executed.update(layer_calls_per_forward="at most " + str(program.layer_calls),
                    layer_calls=sum(calls), passes_per_sequence={str(passes): count for passes, count in used.items()},
                    mean_passes=round(sum(passes * count for passes, count in used.items()) / sum(used.values()), 3))
            retrofit.remove()
        peak = torch.cuda.max_memory_allocated() / 1024**3
    record = {
        "schema_version": 1,
        "purpose": "Benchmark run of a parent or a retrofit design. A measurement, not a finding.",
        "model": retrofit_identity(args.profile, args.method), "profile": args.profile,
        "design": args.method, "passes": args.passes, "program": program.describe() if program else None,
        "settings": {"tasks": args.tasks, "fewshot": args.fewshot, "limit": args.limit,
            "batch_size": args.batch_size, "chat_template": args.chat_template, "precision": args.precision,
            "seed": args.seed, "max_length": args.max_length, "rolling_context": args.rolling_context},
        "datasets": datasets,
        "harness": {name: importlib.metadata.version(name) for name in ("lm_eval", "datasets")}
            | {key: results["config"].get(key) for key in ("random_seed", "numpy_seed", "torch_seed", "fewshot_seed")},
        "runtime": info, "executed": executed, "seconds": round(seconds, 1), "peak_allocated_GiB": round(peak, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(), "groups": bench.group_scores(results),
        "tasks": bench.compact(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for task, entry in record["tasks"].items():
        print(f"{record['model']} passes={args.passes} {task}: {entry['scores']} over {entry['items_run']} items "
            f"in {seconds:.0f}s", flush=True)
    for group, scores in record["groups"].items():
        print(f"{record['model']} passes={args.passes} group {group}: {scores}", flush=True)


LATER_SETTINGS = ("seed", "max_length", "rolling_context")


def with_later_settings(record: dict) -> dict:
    """Records written before a setting existed ran with its default, which is None."""
    for key in LATER_SETTINGS:
        record["settings"].setdefault(key, None)
    return record


def pair_tasks(parent: dict, design: dict) -> dict:
    """Outcome counts for every task and every metric the two records share. A metric that
    does not score items 0 or 1, such as a perplexity, gets the two task scores instead."""
    return {task: {metric: bench.compared(entry, design["tasks"][task], metric) for metric in entry["metrics"]}
        for task, entry in parent["tasks"].items()}


def pair_totals(pairs: dict) -> dict:
    """The outcome counts summed over the tasks of a run, for a suite of many tasks such as MMLU."""
    totals: dict = {}
    for metrics in pairs.values():
        for metric, counted in metrics.items():
            if "both" in counted:
                total = totals.setdefault(metric, dict.fromkeys(("items", *bench.OUTCOMES, "parent_correct", "design_correct"), 0))
                for key in total:
                    total[key] += counted[key]
    return totals


def compare(args) -> None:
    parent, design = (json.loads(path.read_text(encoding="utf-8")) for path in (args.parent, args.design))
    across = parent["runtime"]["parent"] != design["runtime"]["parent"]
    if across and not args.across_checkpoints:
        raise SystemExit("The two runs used different checkpoints; pass --across-checkpoints if that is the comparison")
    for record in (parent, design):
        with_later_settings(record)
    for key in SAME:
        if parent[key] != design[key]:
            raise SystemExit(f"The two runs differ in {key}: {parent[key]} versus {design[key]}")
    pairs = pair_tasks(parent, design)
    summary = {"parent": {"model": parent["model"], "passes": parent["passes"]},
        "design": {"model": design["model"], "passes": design["passes"]}, "settings": parent["settings"],
        "the_two_runs_use_different_checkpoints": across, "tasks": pairs}
    if len(pairs) > 1:
        summary["all_tasks"] = pair_totals(pairs)
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


def summarize(args) -> None:
    """Write the compact, tracked summary of a directory of run records, without per-item data."""
    import hashlib
    paths = sorted(args.input.resolve().glob("*.json"))
    if not paths:
        raise SystemExit("No run records found")
    records = {path.name: with_later_settings(json.loads(path.read_text(encoding="utf-8"))) for path in paths}
    runs = []
    for path in paths:
        record = records[path.name]
        entry = {"record": path.name, "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{key: record[key] for key in ("model", "design", "passes", "settings", "datasets", "harness",
                "executed", "seconds", "peak_allocated_GiB", "completed_at")},
            "versions": record["runtime"]["versions"], "gpu": record["runtime"]["gpu"],
            "groups": record.get("groups", {}), "tasks": {}}
        for task, result in record["tasks"].items():
            entry["tasks"][task] = {"scores": result["scores"], "items_run": result["items_run"],
                "correct": {metric: sum(1 for item in result["items"] if item[metric] == 1.0)
                    for metric in result["metrics"] if bench.is_binary(result, metric)},
                "prompts_sha256": result["prompts_sha256"]}
        if len(entry["tasks"]) > 1:
            entry["all_tasks"] = {"items_run": sum(task["items_run"] for task in entry["tasks"].values()),
                "correct": {metric: sum(task["correct"].get(metric, 0) for task in entry["tasks"].values())
                    for metric in sorted({m for task in entry["tasks"].values() for m in task["correct"]})}}
        runs.append(entry)
    # A design is paired with the plain runs of its own checkpoint. A checkpoint trained to loop
    # has none, so it is paired with every plain run under the same settings, each named.
    pairs = []
    for name, record in records.items():
        if record["design"] is None:
            continue
        trained = "gated" in record["runtime"]
        for other, parent in records.items():
            same = parent["runtime"]["parent"] == record["runtime"]["parent"]
            if parent["design"] is None and (same or trained) and all(parent[key] == record[key] for key in SAME):
                counted = pair_tasks(parent, record)
                pairs.append({"parent_record": other, "design_record": name, "parent_model": parent["model"],
                    "design": record["design"], "passes": record["passes"],
                    "the_two_runs_use_different_checkpoints": not same, "tasks": counted,
                    **({"all_tasks": pair_totals(counted)} if len(counted) > 1 else {})})

    def recirculation(record: dict, passes: int) -> bool:
        return (record.get("program") or {}).get("kind") == "recirculate" and record["passes"] == passes

    # A recirculation is also paired with its own baseline: the parent processed one
    # token at a time, which is a one-pass run of any recirculation design.
    token_pairs = [{"baseline_record": other, "design_record": name, "design": record["design"],
            "the_parent_in_these_counts_is": "the token-by-token baseline", "tasks": pair_tasks(baseline, record)}
        for name, record in records.items() if recirculation(record, 2)
        for other, baseline in records.items()
        if recirculation(baseline, 1) and all(baseline[key] == record[key] for key in SAME)]
    summary = {"schema_version": 1,
        "purpose": "Benchmark-runner validation runs; measurements for checking the tool, not findings",
        "source_directory": args.input.as_posix(), "note": args.note, "runs": runs, "pairs": pairs,
        "token_by_token_pairs": token_pairs}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summarized {len(runs)} runs, {len(pairs)} parent-design pairs and {len(token_pairs)} token-by-token pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    root = {"type": Path, "default": Path.home() / "llm/recurrent"}
    command = commands.add_parser("fetch")
    command.add_argument("--tasks", nargs="+", choices=sorted(bench.PINS["tasks"]))
    command.add_argument("--root", **root)
    command.set_defaults(handler=fetch)
    command = commands.add_parser("run")
    command.add_argument("--profile", required=True,
        choices=sorted(name for name, entry in profiles().items() if entry.get("family") in FAMILIES))
    command.add_argument("--method", help="A design such as block[15:19]/euler, "
        "recirculate[18to9]/convex,alpha=0.07 or gated[2:27]/halt,threshold=0.6; leave out for the parent")
    command.add_argument("--passes", type=int, default=1)
    command.add_argument("--tasks", nargs="+", required=True, choices=sorted(bench.PINS["tasks"]))
    command.add_argument("--fewshot", type=int, help="Examples per prompt; default is each task's own")
    command.add_argument("--limit", type=int, help="Run only the first items of each task")
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--chat-template", action="store_true")
    command.add_argument("--seed", type=int, help="Set all four harness seeds; default is the harness's own")
    command.add_argument("--max-length", type=int, help="Context length in tokens: longer inputs are cut from the "
        "left, and it is the window of a perplexity task; default is the model configuration's")
    command.add_argument("--rolling-context", type=int, help="Perplexity tasks: score every token with at least this "
        "many tokens of context, in overlapping windows; default is the harness's disjoint windows. Needs --max-length")
    command.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    command.add_argument("--root", **root)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(handler=run)
    command = commands.add_parser("compare")
    command.add_argument("parent", type=Path)
    command.add_argument("design", type=Path)
    command.add_argument("--output", type=Path)
    command.add_argument("--across-checkpoints", action="store_true",
        help="Allow two different checkpoints, such as a trained loop and the parent it was trained from")
    command.set_defaults(handler=compare)
    command = commands.add_parser("summarize")
    command.add_argument("input", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--note", help="A caveat to store with the summary")
    command.set_defaults(handler=summarize)
    args = parser.parse_args()
    try:
        args.handler(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
