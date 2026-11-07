"""Plan, preview, freeze, execute, and analyze the Sonnet plain/0-1 confirmation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import subprocess

from eval_core.artifacts import file_sha256
from eval_core.models import TokenUsage
from eval_core.workspace import git_provenance, load_workspace
from recurrent_steg.confirmation.analysis import DECISION_RULE
from recurrent_steg.confirmation.design import (CONDITION, PROTOCOL, check_jobs, control_carriers, control_jobs,
    decoding_jobs, encoding_jobs, judge_jobs, load_design, prompt_identity, shown)
from recurrent_steg.confirmation.execution import PHASES
from recurrent_steg.reference.cli import load_api_keys
from recurrent_steg.reference.design import decoder_prompt, encoder_prompt

DEFAULT_CONFIG = Path("experiments/recurrent_steganography/configs/confirmation-v1.toml")
TYPICAL_OUTPUT = {"judge-controls": 40, "encode": 45, "decoder-check": 10, "decode": 10, "judge": 40}
# Structured outputs add a schema prompt; the amendment measured about 365 input tokens per decode.
ANSWER_ONLY_INPUT = 370
PLACEHOLDER = "This placeholder stands in for an ordinary paraphrase of about the same length as a real carrier."


def source_paths(root: Path) -> list[Path]:
    paths = sorted((root / "src/recurrent_steg/confirmation").glob("*.py"))
    paths += sorted((root / "src/recurrent_steg/reference").glob("*.py"))
    paths += [root / "src/recurrent_steg/reference_amendment/provider.py", root / "src/recurrent_steg/__init__.py"]
    paths += sorted((root / "src/eval_core").rglob("*.py"))
    return paths + [root / "pyproject.toml"]


def plan(config_path: Path) -> dict:
    config_path = Path(config_path).resolve()
    config, cases, models, data_path = load_design(config_path)
    root = load_workspace(config_path).root
    simulated = [dict(job, carrier=PLACEHOLDER) for job in encoding_jobs(config, cases)]
    phases = {
        "judge-controls": (control_jobs(config, cases), models[config["judge"]], config["judge_tokens"]),
        "encode": (encoding_jobs(config, cases), models[config["subject"]], config["encoder_tokens"]),
        "decoder-check": (check_jobs(config, simulated, cases), models[config["subject"]], config["decoder_tokens"]),
        "decode": (decoding_jobs(config, simulated, cases), models[config["subject"]], config["decoder_tokens"]),
        "judge": (judge_jobs(config, simulated), models[config["judge"]], config["judge_tokens"]),
    }
    calls, estimate, bound = {}, Decimal(), Decimal()
    for phase, (jobs, model, cap) in phases.items():
        inputs = [ANSWER_ONLY_INPUT if phase in ("decoder-check", "decode") else max(1, len(job["prompt"]) // 4)
            for job in jobs]
        calls[phase] = dict(calls=len(jobs), model=model.id, max_output_tokens=cap)
        estimate += sum((model.rate.actual_cost(TokenUsage(tokens, TYPICAL_OUTPUT[phase])) for tokens in inputs), Decimal())
        bound += sum((model.rate.reserve_cost(tokens, cap) for tokens in inputs), Decimal())
    cent = Decimal("0.01")
    return dict(schema_version=1, experiment_id="recurrent_steganography", protocol=PROTOCOL, role=config["role"],
        config=config, config_path=config_path.relative_to(root).as_posix(), config_sha256=file_sha256(config_path),
        data_sha256=file_sha256(data_path),
        source_sha256={path.relative_to(root).as_posix(): file_sha256(path) for path in source_paths(root)},
        git=git_provenance(root), subject=config["subject"], judge=config["judge"], condition=CONDITION,
        prompt_fingerprints=prompt_identity(),
        option_orders={case["id"]: case["option_order"] for case in cases if case["split"] == "development"},
        controls=[{key: value for key, value in row.items() if key != "message"} for row in control_carriers(cases)],
        calls=calls, total_calls=sum(entry["calls"] for entry in calls.values()),
        estimated_cost_usd=str(estimate.quantize(cent)), all_calls_at_output_cap_usd=str(bound.quantize(cent)),
        run_budget_usd=config["run_budget_usd"], decision_rule=DECISION_RULE, ready_for_generation=True)


def verify_freeze(run_dir: Path) -> dict:
    frozen = json.loads((Path(run_dir) / "plan.json").read_text(encoding="utf-8"))
    if frozen.get("protocol") != PROTOCOL:
        raise ValueError("This run directory does not hold a confirmation plan")
    current = plan(load_workspace().root / frozen["config_path"])
    for key in ("config_sha256", "data_sha256", "source_sha256", "prompt_fingerprints", "option_orders"):
        if current[key] != frozen[key]:
            raise ValueError("Frozen confirmation identity changed: " + key)
    return frozen


def freeze(config_path: Path, output: Path | None) -> tuple[Path, dict]:
    result = plan(config_path)
    root = load_workspace(config_path).root
    data_file = ((root / result["config_path"]).parent / result["config"]["data"]).resolve()
    identity = list(result["source_sha256"]) + [result["config_path"], data_file.relative_to(root).as_posix()]
    changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *identity],
        cwd=root, text=True, encoding="utf-8")
    if changed.strip():
        raise ValueError("Commit the confirmation source, configuration, and data before freezing")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output or root / "artifacts/runs/recurrent_steganography" / f"{stamp}-confirmation"
    if destination.exists():
        raise ValueError("Preserve existing evidence; choose another output directory")
    destination.mkdir(parents=True)
    (destination / "plan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return destination, result


def preview(config_path: Path) -> str:
    result = plan(config_path)
    config, cases, models, _ = load_design(Path(config_path).resolve())
    example = next(case for case in cases if case["split"] == "development")
    lines = [f"# Draft for review: {PROTOCOL}", "",
        "Nothing here has been frozen or run. One pre-registered cell: Claude Sonnet 5.5 "
        f"(`{models[config['subject']].model}`, thinking `{models[config['subject']].thinking_type}`) writes and "
        f"decodes; `{models[config['judge']].model}` judges. Prompts are the frozen plain/0-1 reference prompts; the "
        "decoder's reply is restricted to one of the two options shown.", "",
        f"Calls: {result['total_calls']}. Estimated cost: ${result['estimated_cost_usd']}. Proposed ceiling: "
        f"${result['run_budget_usd']}.", "", "## Decision rule", "", DECISION_RULE, "",
        "## Prompts (example cover)", "", "```text", encoder_prompt(example["message"], CONDITION, "1"), "```", "",
        "```text", decoder_prompt("<the encoder's paraphrase>", CONDITION, shown(example)), "```", "",
        "## Development covers", "", "| ID | Family | Decoder order | Cover |", "| --- | --- | --- | --- |"]
    lines += [f"| {c['id']} | {c['family']} | {c['option_order']} | {c['message']} |"
        for c in cases if c["split"] == "development"]
    lines += ["", "## Judge calibration covers", "", "| ID | Family | Cover |", "| --- | --- | --- |"]
    lines += [f"| {c['id']} | {c['family']} | {c['message']} |" for c in cases if c["split"] == "judge_calibration"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m recurrent_steg.confirmation",
        description="Pre-registered confirmation of the Sonnet 5.5 plain/0-1 cell")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "preview", "freeze"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name in ("preview", "freeze"):
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
            from recurrent_steg.confirmation.execution import execute
            print(json.dumps(execute(args.run.resolve(), args.phase, live=True), indent=2))
        elif args.command == "analyze":
            from recurrent_steg.confirmation.analysis import analyze
            result = analyze(args.run.resolve())
            print(json.dumps({key: result[key] for key in ("confirmed", "primary", "secondary", "total_spend_usd")},
                indent=2))
        elif args.command == "preview":
            text = preview(args.config)
            if args.output:
                args.output.write_text(text, encoding="utf-8")
            else:
                print(text)
        elif args.command == "freeze":
            destination, result = freeze(args.config, args.output)
            print(json.dumps(dict(run_dir=str(destination), total_calls=result["total_calls"],
                estimated_cost_usd=result["estimated_cost_usd"]), indent=2))
        else:
            result = plan(args.config)
            if args.command == "validate":
                result = {key: result[key] for key in ("protocol", "subject", "judge", "calls", "total_calls",
                    "estimated_cost_usd", "all_calls_at_output_cap_usd", "run_budget_usd")}
            print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, "error: " + str(exc) + "\n")
