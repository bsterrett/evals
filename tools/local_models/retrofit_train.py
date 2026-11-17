"""Train a gate over a frozen parent's block loop, and measure the result (WSL).

  fetch      download the pinned training text; the only step that uses the network
  prepare    tokenize the text with the parent's tokenizer and pack it into rows, on the CPU
  train      train for one segment, resuming the run folder's checkpoint if there is one
  falling    exit with 0 if a run's held-out loss is still falling, and with 3 if it is not
  measure    loss by number of passes through the runner: parent, untrained loop, trained loop
  summarize  write the compact, tracked summary of a directory of run records

`train` holds the GPU lock for one segment of at most `--minutes` and then
exits, so other GPU jobs can take a turn. Run it again with the same settings to
continue: a run folder holds one run, and refuses other settings. `--tokens` is
the most training text the run may have read when the command returns, so
raising it extends a finished run.

During training the loss after each pass is computed by hand on held-out rows,
every `--measure-every` tokens, and the loop is saved beside it. Scored passes
are drawn evenly, so training minimises that loss averaged over the passes. If
the average rises above the untrained loop's, the run is diverging: `train`
stops it with exit code 4 and leaves the checkpoint at the last good point. Use a
lower learning rate in a new run folder. `measure` then
puts a saved loop through the toolkit's runner, the path the LoopUS checkpoint
was checked on, and compares it with the same loop before training and with the
plain parent. A loop took if, at every number of passes from 2 up, its loss is
below the untrained loop's. That shows training did something. It does not show
that looping improves on the parent.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import identity, profiles
from eval_core.local.worker import gpu_lock

TEXT = "fineweb-edu-2025-26-000"
HEADROOM_BYTES = 1024**3
PROGRESS_BATCHES = 50
DIVERGED = 4  # exit code of `train` when it stops a run that is getting worse


def lock(args):
    """The shared GPU lock, taken for GPU work only."""
    return gpu_lock(args.root) if args.placement == "gpu" else contextlib.nullcontext()


def tokens_folder(args, text: str, profile: str, context: int) -> Path:
    return args.tokens_folder or args.root / "training" / text / f"{profile}-c{context}"


def place(args) -> None:
    """Cap PyTorch's memory below what is free, so a batch that does not fit raises an error instead
    of spilling into shared system memory; on the CPU, set the number of threads."""
    import torch
    if args.placement == "gpu":
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction((free - HEADROOM_BYTES) / total)
    else:
        torch.set_num_threads(args.threads)


def fetch(args) -> None:
    from recurrent_retrofit import training
    path = training.fetch_text(args.root, args.text)
    print(json.dumps({"text": args.text, **training.text_pin(args.text), "path": str(path), "verified": True}, indent=2), flush=True)


def prepare(args) -> None:
    import os
    root = args.root.resolve()
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from recurrent_retrofit import training
    settings = profiles()[args.profile]
    if settings.get("family") != "qwen3":
        raise SystemExit("prepare reads the tokenizer of a plain parent")
    snapshot = snapshot_download(settings["repo_id"], revision=settings["revision"],
        cache_dir=str(root / "cache/huggingface/hub"), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    separator = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if separator is None or separator == tokenizer.unk_token_id:
        raise SystemExit("The tokenizer has no <|endoftext|> token to put between documents")
    pin = training.text_pin(args.text)
    folder = tokens_folder(args, args.text, args.profile, args.context)
    started = time.perf_counter()
    manifest = training.prepare_text(training.text_file(root, args.text), pin["field"], tokenizer, folder,
        context=args.context, separator=separator, train_rows=args.train_tokens // args.context,
        heldout_rows=args.heldout_rows, facts={"text": args.text, "source": f"{pin['dataset']}@{pin['revision']}",
            "file": pin["file"], "file_sha256": pin["sha256"], "tokenizer": identity(args.profile),
            "between_documents": "<|endoftext|>"})
    print(json.dumps({**manifest, "folder": str(folder), "seconds": round(time.perf_counter() - started, 1)}, indent=2), flush=True)


def train(args) -> None:
    import numpy
    import torch
    from recurrent_retrofit import training
    from recurrent_retrofit.loading import load_parent
    unit = tuple(int(part) for part in args.unit.split(":"))
    rows, heldout, text = training.load_tokens(tokens_folder(args, args.text, args.profile, args.context))
    if text["tokenizer"] != identity(args.profile):
        raise SystemExit("The rows were tokenized for another parent")
    settings = {"profile": args.profile, "parent": identity(args.profile), "precision": args.precision,
        "placement": args.placement, "unit": list(unit), "gate_rank": args.gate_rank, "passes": args.passes,
        "supervised": args.supervised, "batch": args.batch, "context": text["context"],
        "learning_rate": args.learning_rate, "warmup_updates": args.warmup_updates, "accumulate": args.accumulate,
        "penalty": args.penalty, "clip": args.clip, "seed": args.seed, "open_bias": training.OPEN_BIAS,
        "monitor_rows": args.monitor_rows, "measure_every_tokens": args.measure_every,
        "text": {"name": text["text"], "source": text["source"], "file": text["file"],
            "train_sha256": text["train"]["sha256"], "heldout_sha256": text["heldout"]["sha256"]}}
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    stored = run / "settings.json"
    if stored.exists():
        if json.loads(stored.read_text(encoding="utf-8")) != settings:
            raise SystemExit(f"{run} holds a run with other settings; use another run folder")
    else:
        stored.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    per_batch = args.batch * text["context"]
    target = min(args.tokens // per_batch, len(rows) // args.batch) // args.accumulate * args.accumulate
    every = max(args.accumulate, args.measure_every // per_batch // args.accumulate * args.accumulate)
    with lock(args):
        place(args)
        model, _, info = load_parent(args.profile, args.root, precision=args.precision, placement=args.placement)
        device = next(model.parameters()).device
        trainer = training.Trainer(model, unit, passes=args.passes, supervised=args.supervised,
            learning_rate=args.learning_rate, warmup_updates=args.warmup_updates, accumulate=args.accumulate,
            gate_rank=args.gate_rank, penalty=args.penalty, clip=args.clip, seed=args.seed)
        parent_sha256 = training.fingerprint(model)
        checkpoint, log, earlier = run / "checkpoint.pt", run / "log.jsonl", 0.0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            if saved["parent_sha256"] != parent_sha256:
                raise SystemExit("The parent's weights differ from those this run began with")
            trainer.restore(saved["trainer"])
            earlier = saved["seconds"]
        order = training.row_order(args.seed, len(rows))
        monitor = heldout[:args.monitor_rows]
        recent: dict[int, list[float]] = {}
        started = time.perf_counter()
        began_at = trainer.batches

        def seconds() -> float:
            return earlier + time.perf_counter() - started

        def save() -> None:
            temporary = run / "checkpoint.tmp"
            torch.save({"trainer": trainer.state(), "parent_sha256": parent_sha256, "seconds": seconds()}, temporary)
            temporary.replace(checkpoint)

        def measure_point() -> None:
            """Measure on held-out rows, save the loop as it is now, and save the run. Stop the run
            if training has made the loss it minimises worse than the untrained loop's."""
            tokens = trainer.batches * per_batch
            entry = {"tokens": tokens, "batches": trainer.batches, "updates": trainer.updates,
                "heldout": training.loss_by_passes(trainer.parts, trainer.loop, monitor, args.passes, args.batch),
                "train_loss_by_scored_pass": {str(index): round(sum(values) / len(values), 5)
                    for index, values in sorted(recent.items())},
                "seconds": round(seconds(), 1)}
            entry["heldout"]["mean_over_passes"] = round(sum(entry["heldout"]["loss_after_pass"]) / args.passes, 5)
            untrained = entries_of(run)[0]["heldout"]["mean_over_passes"] if log.exists() else None
            diverged = untrained is not None and entry["heldout"]["mean_over_passes"] > untrained
            if diverged:
                entry["stopped"] = f"the mean held-out loss over the passes is above the untrained loop's {untrained}"
            recent.clear()
            training.save_loop(run / "loops" / str(tokens), trainer.loop, {"parent": settings["parent"],
                "parent_profile": args.profile, "parent_sha256": parent_sha256, "precision": args.precision,
                "tokens_seen": tokens, "updates": trainer.updates, "settings": settings,
                "saved_at": datetime.now(timezone.utc).isoformat()})
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            if diverged:
                # The checkpoint stays at the last good point. A lower learning rate needs a new run folder.
                raise SystemExit(DIVERGED)
            save()

        print(json.dumps({"run": str(run), "resumed_at_batch": trainer.batches, "target_batches": target,
            "tokens_per_batch": per_batch, "measure_every_batches": every, "runtime": info["parent"]}), flush=True)
        if trainer.batches == 0 and not log.exists():
            measure_point()
        while trainer.batches < target:
            chosen = numpy.sort(order[trainer.batches * args.batch:(trainer.batches + 1) * args.batch])
            ids = torch.from_numpy(rows[chosen].astype(numpy.int64)).to(device)
            for row in trainer.step(ids):
                recent.setdefault(row["pass"], []).append(row["loss"])
            if trainer.batches % PROGRESS_BATCHES == 0:
                rate = (trainer.batches - began_at) * per_batch / (time.perf_counter() - started)
                print(f"  batch {trainer.batches}/{target}, {rate:,.0f} tokens per second with measuring", flush=True)
            between_updates = trainer.batches % args.accumulate == 0
            out_of_time = between_updates and time.perf_counter() - started > args.minutes * 60
            if trainer.batches % every == 0 or trainer.batches == target:
                measure_point()
            elif out_of_time:
                save()
            if out_of_time:
                break
        if training.fingerprint(model) != parent_sha256:
            raise SystemExit("The parent's weights changed during training")
        tokens = trainer.batches * per_batch
        print(json.dumps({"done": trainer.batches >= target, "tokens": tokens, "seconds": round(seconds(), 1),
            "loop": str(run / "loops" / str(tokens)) if trainer.batches >= target else None,
            "parent_weights_unchanged": True}), flush=True)


def entries_of(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def falling(args) -> None:
    """Whether the held-out loss at the trained number of passes fell by more than `--margin` over the
    run's last `--back` tokens."""
    entries = entries_of(args.run)
    last = entries[-1]
    earlier = [entry for entry in entries if entry["tokens"] <= last["tokens"] - args.back]
    if not earlier:
        raise SystemExit("The run is too short to say")
    reference = earlier[-1]
    drop = reference["heldout"]["loss_after_pass"][-1] - last["heldout"]["loss_after_pass"][-1]
    result = {"at_tokens": [reference["tokens"], last["tokens"]],
        "loss": [reference["heldout"]["loss_after_pass"][-1], last["heldout"]["loss_after_pass"][-1]],
        "drop": round(drop, 5), "margin": args.margin, "falling": drop > args.margin}
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["falling"] else 3)


def measure(args) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gated_check import passage_loss, passages
    from recurrent_retrofit import training
    from recurrent_retrofit.gated import Loop
    from recurrent_retrofit.loading import load_parent
    from recurrent_retrofit.program import GatedLoop
    from recurrent_retrofit.runner import install
    loop, manifest = training.load_loop(args.loop)
    settings, unit = manifest["settings"], tuple(manifest["unit"])
    profile = manifest["parent_profile"]
    _, heldout, text = training.load_tokens(tokens_folder(args, settings["text"]["name"], profile, settings["context"]))
    if text["heldout"]["sha256"] != settings["text"]["heldout_sha256"]:
        raise SystemExit("The held-out rows are not the ones this loop's run held out")
    with lock(args):
        place(args)
        model, tokenizer, info = load_parent(profile, args.root, precision=args.precision, placement=args.placement)
        layers = info["parent_layers"]
        texts, text_facts = passages(args.root, tokenizer, args.passages)
        sets = {"passages": texts, "heldout": [row.tolist() for row in heldout[:args.heldout_rows]]}
        record = {"loop": manifest, "loop_folder": str(args.loop), "runtime": info, "passage_text": text_facts,
            "heldout_text": {"rows": len(sets["heldout"]), "context": text["context"], "source": text["source"],
                "file": text["file"], "documents": text["heldout"]["documents"]},
            "parent_weights_are_those_trained_over": training.fingerprint(model) == manifest["parent_sha256"]
                if args.precision == manifest["precision"] else None,
            "parent": {name: passage_loss(model, rows) for name, rows in sets.items()}, "loops": {}}
        print(json.dumps({"parent": record["parent"]}), flush=True)
        untrained = Loop(manifest["hidden"], manifest["gate_rank"], unit, head=manifest["head"],
            trained_passes=manifest["trained_passes"])
        training.open_gate(untrained, settings["open_bias"])
        for label, candidate in (("untrained", untrained), ("trained", loop)):
            training.attach(model, candidate, manifest, info["parent"])
            record["loops"][label] = []
            for passes in args.passes:
                design = GatedLoop(layers, *unit, passes)
                entry = {"design": design.spec, "passes": passes}
                with install(model, design) as installed:
                    for name, rows in sets.items():
                        installed.take_calls()
                        entry[name] = passage_loss(model, rows)
                        installed.check_calls(installed.take_calls(), len(rows))
                record["loops"][label].append(entry)
                print(json.dumps({label: entry}), flush=True)
            del model.gated_loop
        # The trainer's own by-hand forward pass on the same held-out rows, to compare with the runner's.
        device = next(model.parameters()).device
        record["by_hand_on_heldout"] = training.loss_by_passes(training.Parts(model, unit), loop.to(device),
            heldout[:args.heldout_rows], max(args.passes), 1)
        trained, before = record["loops"]["trained"], record["loops"]["untrained"]
        record["took"] = {name: all(new[name]["mean_loss"] < old[name]["mean_loss"]
            for new, old in zip(trained, before, strict=True) if new["passes"] >= 2) for name in sets}
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"train-measure-{profile}-{unit[0]}-{unit[1]}-{manifest['tokens_seen']}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"took": record["took"], "by_hand_on_heldout": record["by_hand_on_heldout"]}), flush=True)
    print(f"Saved {destination}", flush=True)


def summarize(args) -> None:
    """The compact, tracked summary of a directory holding runs' `<run>-settings.json` and
    `<run>-log.jsonl` copies and `measure` records."""
    import hashlib
    folder = args.input.resolve()

    def read(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    runs = []
    for path in sorted(folder.glob("*-settings.json")):
        name = path.name[:-len("-settings.json")]
        log = [json.loads(line) for line in (folder / f"{name}-log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        runs.append({"run": name, "settings": read(path), "seconds_training_and_measuring": log[-1]["seconds"],
            "stopped": log[-1].get("stopped"),
            "heldout_during_training": [{"tokens": entry["tokens"], "updates": entry["updates"], **entry["heldout"]} for entry in log]})
    measurements = []
    for path in sorted(folder.glob("train-measure-*.json")):
        record, runtime = read(path), read(path)["runtime"]
        measurements.append({"record": path.name, "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "loop": {name: record["loop"][name] for name in ("parent", "unit", "gate_rank", "trained_passes", "tokens_seen",
                "updates", "precision", "parent_sha256", "tensors_sha256")},
            "precision": runtime["precision"], "versions": runtime["versions"], "gpu": runtime["gpu"],
            "passage_text": record["passage_text"], "heldout_text": record["heldout_text"],
            "parent_weights_are_those_trained_over": record["parent_weights_are_those_trained_over"],
            "parent": record["parent"], "loops": record["loops"], "by_hand_on_heldout": record["by_hand_on_heldout"],
            "took": record["took"], "completed_at": record["completed_at"]})
    if not runs and not measurements:
        raise SystemExit("No run records found")
    summary = {"schema_version": 1, "purpose": "Gate training over a frozen parent; validates the trainer, not the method",
        "source_directory": args.input.as_posix(), "note": args.note, "runs": runs, "measurements": measurements}
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summarized {len(runs)} runs and {len(measurements)} measurements", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plain = [name for name, settings in profiles().items() if settings.get("family") == "qwen3"]

    def shared(command, placement: bool = True) -> None:
        command.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
        command.add_argument("--tokens-folder", type=Path, help="Where the packed rows are; default is under the root")
        if placement:
            command.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
            command.add_argument("--placement", choices=["gpu", "cpu"], default="gpu")
            command.add_argument("--threads", type=int, default=16, help="CPU threads, for --placement cpu")

    command = commands.add_parser("fetch", help="Download the pinned training text")
    command.add_argument("--text", default=TEXT)
    command.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    command.set_defaults(handler=fetch)

    command = commands.add_parser("prepare", help="Tokenize the text and pack it into rows")
    command.add_argument("--profile", required=True, choices=plain)
    command.add_argument("--text", default=TEXT)
    command.add_argument("--context", type=int, default=1024)
    command.add_argument("--train-tokens", type=int, required=True)
    command.add_argument("--heldout-rows", type=int, default=256)
    shared(command, placement=False)
    command.set_defaults(handler=prepare)

    command = commands.add_parser("train", help="Train for one segment")
    command.add_argument("--profile", required=True, choices=plain)
    command.add_argument("--unit", required=True, help="start:end of the looped layers")
    command.add_argument("--run", type=Path, required=True, help="The run's folder: settings, log, checkpoint and saved loops")
    command.add_argument("--tokens", type=int, required=True, help="The most training text the run may have read")
    command.add_argument("--minutes", type=float, default=45, help="Stop this segment after this long")
    command.add_argument("--text", default=TEXT)
    command.add_argument("--context", type=int, default=1024)
    command.add_argument("--passes", type=int, default=8)
    command.add_argument("--supervised", type=int, default=2, help="How many of the passes are scored")
    command.add_argument("--batch", type=int, default=4)
    command.add_argument("--accumulate", type=int, default=2, help="Batches per optimizer update")
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--warmup-updates", type=int, default=50)
    command.add_argument("--gate-rank", type=int, help="Default is a sixteenth of the hidden size, as LoopUS")
    command.add_argument("--penalty", type=float, default=1.0, help="Weight of the penalty when a pass makes the loss worse")
    command.add_argument("--clip", type=float, default=1.0, help="Largest gradient norm; 0 for no clipping")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--measure-every", type=int, default=2_000_000, help="Tokens between held-out measurements")
    command.add_argument("--monitor-rows", type=int, default=32, help="Held-out rows measured during training")
    shared(command)
    command.set_defaults(handler=train)

    command = commands.add_parser("falling", help="Is the held-out loss still falling?")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--back", type=int, default=4_000_000, help="Tokens to look back")
    command.add_argument("--margin", type=float, default=0.005)
    command.set_defaults(handler=falling)

    command = commands.add_parser("measure", help="Loss by number of passes, through the runner")
    command.add_argument("--loop", type=Path, required=True, help="A saved loop's folder")
    command.add_argument("--passes", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    command.add_argument("--passages", type=int, default=200)
    command.add_argument("--heldout-rows", type=int, default=64)
    command.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/local-runtime"
        / datetime.now(timezone.utc).strftime("retrofit-train-%Y%m%dT%H%M%SZ"))
    shared(command)
    command.set_defaults(handler=measure)

    command = commands.add_parser("summarize", help="Write the compact summary of a directory of run records")
    command.add_argument("input", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--note", help="A caveat to store with the summary")
    command.set_defaults(handler=summarize)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
