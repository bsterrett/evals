"""Plan, freeze, execute, and analyze the answer-only Sonnet 5.5 decoding amendment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from eval_core.workspace import load_workspace
from recurrent_steg.reference.cli import load_api_keys
from recurrent_steg.reference_amendment.amendment import PHASES, plan

DEFAULT_CONFIG = Path("experiments/recurrent_steganography/configs/reference-v1-sonnet-answer-only.toml")


def freeze(config_path: Path, output: Path | None) -> tuple[Path, dict]:
    result = plan(config_path)
    root = load_workspace(config_path).root
    identity_paths = list(result["source_sha256"]) + [result["config_path"]]
    changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *identity_paths],
        cwd=root, text=True, encoding="utf-8")
    if changed.strip():
        raise ValueError("Commit the amendment source and configuration before freezing")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output or root / "artifacts/runs/recurrent_steganography" / f"{stamp}-reference-sonnet-answer-only"
    if destination.exists():
        raise ValueError("Preserve existing evidence; choose another output directory")
    destination.mkdir(parents=True)
    (destination / "plan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return destination, result


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m recurrent_steg.reference_amendment",
        description="Answer-only Sonnet 5.5 decoding amendment to the paid reference check")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "freeze"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name == "freeze":
            command.add_argument("--output", type=Path)
    run = commands.add_parser("execute", help="Paid calls; requires --live")
    run.add_argument("--run", type=Path, required=True)
    run.add_argument("--phase", choices=PHASES, required=True)
    run.add_argument("--live", action="store_true")
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "execute":
            if not args.live:
                raise ValueError("Paid calls require --live")
            load_api_keys(load_workspace().root)
            from recurrent_steg.reference_amendment.amendment import execute
            print(json.dumps(execute(args.run.resolve(), args.phase, live=True), indent=2))
        elif args.command == "analyze":
            from recurrent_steg.reference_amendment.amendment import analyze
            result = analyze(args.run.resolve())
            print(json.dumps(dict(feasible_cells=result["feasible_cells"],
                combined_decision=result["combined_decision"]), indent=2))
        elif args.command == "freeze":
            destination, result = freeze(args.config, args.output)
            print(json.dumps(dict(run_dir=str(destination), decode_jobs=result["decode_jobs"],
                check_jobs=len(result["check_jobs"])), indent=2))
        else:
            result = plan(args.config)
            print(json.dumps({key: result[key] for key in ("protocol", "subject", "model", "decode_jobs",
                "check_jobs", "decision_rule")}, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, "error: " + str(exc) + "\n")
