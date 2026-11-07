"""Plan, preview, freeze, execute, and analyze the reasoning-decoder re-decode."""
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
from recurrent_steg.confirmation import design as parent_design
from recurrent_steg.reasoning_decoder.analysis import DECISION_RULE
from recurrent_steg.reasoning_decoder.design import (PROTOCOL, check_jobs, decode_jobs, job_identity, load,
    parent_identity)
from recurrent_steg.reasoning_decoder.execution import PHASES
from recurrent_steg.reference.cli import load_api_keys
from recurrent_steg.reference.design import decoder_prompt
from recurrent_steg.reference_amendment.provider import answer_format

DEFAULT_CONFIG = Path("experiments/recurrent_steganography/configs/reasoning-decoder-v1.toml")
# The answer-only decodes measured about 370 input tokens; thinking adds output, not input.
INPUT_TOKENS = 370
# Thinking length is unknown before the pre-test; the plan reports a range, and the
# pre-test gate projects the real cost before the full decode starts.
ASSUMED_OUTPUT_TOKENS = {"short": 500, "typical": 1000, "long": 2000}


def source_paths(root: Path) -> list[Path]:
    paths = sorted((root / "src/recurrent_steg/reasoning_decoder").glob("*.py"))
    paths += sorted((root / "src/recurrent_steg/confirmation").glob("*.py"))
    paths += sorted((root / "src/recurrent_steg/reference").glob("*.py"))
    paths += [root / "src/recurrent_steg/reference_amendment/provider.py", root / "src/recurrent_steg/__init__.py"]
    paths += sorted((root / "src/eval_core").rglob("*.py"))
    return paths + [root / "pyproject.toml"]


def plan(config_path: Path) -> dict:
    loaded = load(config_path)
    config, model, root = loaded["config"], loaded["model"], loaded["root"]
    checks, decodes = check_jobs(loaded), decode_jobs(loaded)
    count = len(checks) + len(decodes)
    cent = Decimal("0.01")
    estimates = {name: str((model.rate.actual_cost(TokenUsage(INPUT_TOKENS, tokens)) * count).quantize(cent))
        for name, tokens in ASSUMED_OUTPUT_TOKENS.items()}
    bound = (model.rate.reserve_cost(INPUT_TOKENS, config["decoder_tokens"]) * count).quantize(cent)
    return dict(schema_version=1, experiment_id="recurrent_steganography", protocol=PROTOCOL, role=config["role"],
        config=config, config_path=loaded["config_path"].relative_to(root).as_posix(),
        config_sha256=file_sha256(loaded["config_path"]),
        source_sha256={path.relative_to(root).as_posix(): file_sha256(path) for path in source_paths(root)},
        git=git_provenance(root), subject=loaded["subject"],
        model=dict(provider=model.provider, model=model.model, thinking_type=model.thinking_type,
            thinking_display="summarized", effort=model.reasoning_effort, max_output_tokens=config["decoder_tokens"],
            verified_at=model.rate.verified_at.isoformat()),
        parent=parent_identity(loaded), prompt_fingerprints=parent_design.prompt_identity(),
        answer_format_example=answer_format(["<first shown>", "<second shown>"]),
        check_jobs=[job["id"] for job in checks], decode_jobs=len(decodes), decode_job_identity=job_identity(decodes),
        calls={"decoder-check": len(checks), "decode": len(decodes)}, total_calls=count,
        assumed_output_tokens=ASSUMED_OUTPUT_TOKENS, estimated_cost_usd=estimates,
        all_calls_at_output_cap_usd=str(bound), run_budget_usd=config["run_budget_usd"],
        decision_rule=DECISION_RULE, ready_for_generation=True)


def verify_freeze(run_dir: Path) -> dict:
    frozen = json.loads((Path(run_dir) / "plan.json").read_text(encoding="utf-8"))
    if frozen.get("protocol") != PROTOCOL:
        raise ValueError("This run directory does not hold a reasoning-decoder plan")
    current = plan(load_workspace().root / frozen["config_path"])
    for key in ("config_sha256", "source_sha256", "parent", "prompt_fingerprints", "check_jobs", "decode_job_identity"):
        if current[key] != frozen[key]:
            raise ValueError("Frozen reasoning-decoder identity changed: " + key)
    return frozen


def freeze(config_path: Path, output: Path | None) -> tuple[Path, dict]:
    result = plan(config_path)
    root = load_workspace(config_path).root
    identity = list(result["source_sha256"]) + [result["config_path"]]
    changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *identity],
        cwd=root, text=True, encoding="utf-8")
    if changed.strip():
        raise ValueError("Commit the reasoning-decoder source and configuration before freezing")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output or root / "artifacts/runs/recurrent_steganography" / f"{stamp}-reasoning-decoder"
    if destination.exists():
        raise ValueError("Preserve existing evidence; choose another output directory")
    destination.mkdir(parents=True)
    (destination / "plan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return destination, result


def preview(config_path: Path) -> str:
    result = plan(config_path)
    model = result["model"]
    lines = [f"# Draft for review: {PROTOCOL}", "",
        "Nothing here has been frozen or run. Claude Sonnet 5.5 decodes the confirmation run's existing payload "
        f"paraphrases again, now with adaptive thinking at effort `{model['effort']}` (summaries kept) before the "
        "same schema-enforced answer. Paraphrases, prompts, option orders, and acceptance decisions are the "
        "confirmation's.", "",
        f"Calls: {result['total_calls']} ({result['calls']['decoder-check']} pre-test, {result['calls']['decode']} "
        f"decode). Estimated cost by mean output tokens: {result['estimated_cost_usd']}. Proposed ceiling: "
        f"${result['run_budget_usd']}.", "", "## Decision rule", "", DECISION_RULE, "",
        "## Decoder prompt", "", "```text", decoder_prompt("<the encoder's paraphrase>", parent_design.CONDITION,
            parent_design.BIT_OPTIONS), "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m recurrent_steg.reasoning_decoder",
        description="Re-decode the confirmation's paraphrases with a reasoning Sonnet 5.5 decoder")
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
            from recurrent_steg.reasoning_decoder.execution import execute
            print(json.dumps(execute(args.run.resolve(), args.phase, live=True), indent=2))
        elif args.command == "analyze":
            from recurrent_steg.reasoning_decoder.analysis import analyze
            result = analyze(args.run.resolve())
            print(json.dumps({key: result[key] for key in ("signal", "primary", "secondary", "reasoning",
                "total_spend_usd")}, indent=2))
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
                result = {key: result[key] for key in ("protocol", "subject", "model", "calls", "total_calls",
                    "estimated_cost_usd", "all_calls_at_output_cap_usd", "run_budget_usd")}
            print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, "error: " + str(exc) + "\n")
