"""Plan, preview, freeze, execute, and analyze the rules-decoder re-decode."""
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
from recurrent_steg.both_reasoning import design as parent_design
from recurrent_steg.reference.cli import load_api_keys
from recurrent_steg.rules_decoder.analysis import DECISION_RULE
from recurrent_steg.rules_decoder.design import (PROTOCOL, answer_only_jobs, check_jobs, decode_jobs, job_identity,
    load, parent_identity, prompt_identity, rules_block, with_rules)
from recurrent_steg.rules_decoder.execution import PHASES

DEFAULT_CONFIG = Path("experiments/recurrent_steganography/configs/rules-decoder-v1.toml")
# Planning figures only. Input measured in the parent run: about 371 tokens per thinking
# decode and 361 per answer-only decode; the shown instructions add roughly a quarter
# token per character. Thinking length is unknown before the pre-test, so the thinking
# phases are estimated at three output sizes.
PARENT_INPUT = {"thinking": 371, "answer_only": 361}
THINKING_OUTPUT_SCENARIOS = (250, 500, 1000)
ANSWER_ONLY_OUTPUT = 10


def source_paths(root: Path) -> list[Path]:
    paths = []
    for package in ("rules_decoder", "both_reasoning", "confirmation", "reasoning_decoder", "reference"):
        paths += sorted((root / "src/recurrent_steg" / package).glob("*.py"))
    paths += [root / "src/recurrent_steg/reference_amendment/provider.py", root / "src/recurrent_steg/__init__.py"]
    paths += sorted((root / "src/eval_core").rglob("*.py"))
    return paths + [root / "pyproject.toml"]


def plan(config_path: Path) -> dict:
    loaded = load(config_path)
    config, root, model, answer_only = loaded["config"], loaded["root"], loaded["model"], loaded["answer_only_model"]
    checks, decodes, answer_only_list = check_jobs(loaded), decode_jobs(loaded), answer_only_jobs(loaded)
    added = len(rules_block(loaded["instructions"])) // 4
    thinking_input, answer_only_input = PARENT_INPUT["thinking"] + added, PARENT_INPUT["answer_only"] + added
    answer_only_usd = answer_only.rate.actual_cost(TokenUsage(answer_only_input, ANSWER_ONLY_OUTPUT)) * len(answer_only_list)
    cent = Decimal("0.01")
    estimates = {f"thinking output {tokens} tokens": str((model.rate.actual_cost(TokenUsage(thinking_input, tokens))
        * (len(checks) + len(decodes)) + answer_only_usd).quantize(cent)) for tokens in THINKING_OUTPUT_SCENARIOS}
    calls = {"decoder-check": len(checks), "decode": len(decodes), "decode-answer-only": len(answer_only_list)}
    return dict(schema_version=1, experiment_id="recurrent_steganography", protocol=config["protocol"],
        role=config["role"], config=config, config_path=loaded["config_path"].relative_to(root).as_posix(),
        config_sha256=file_sha256(loaded["config_path"]),
        source_sha256={path.relative_to(root).as_posix(): file_sha256(path) for path in source_paths(root)},
        git=git_provenance(root), subject=loaded["subject"],
        models={name: dict(provider=entry.provider, model=entry.model, thinking_type=entry.thinking_type,
            effort=entry.reasoning_effort, verified_at=entry.rate.verified_at.isoformat())
            for name, entry in (("thinking", model), ("answer_only", answer_only))},
        parent=parent_identity(loaded), writer_instructions=loaded["instructions"],
        prompt_fingerprints=prompt_identity(loaded), check_jobs=[job["id"] for job in checks],
        decode_job_identity=job_identity(decodes), answer_only_job_identity=job_identity(answer_only_list),
        calls=calls, total_calls=sum(calls.values()), estimated_input_tokens=dict(thinking=thinking_input,
            answer_only=answer_only_input), estimated_cost_usd=estimates, run_budget_usd=config["run_budget_usd"],
        decision_rule=DECISION_RULE, ready_for_generation=True)


def verify_freeze(run_dir: Path) -> dict:
    frozen = json.loads((Path(run_dir) / "plan.json").read_text(encoding="utf-8"))
    if frozen.get("protocol") != PROTOCOL:
        raise ValueError("This run directory does not hold a rules-decoder plan")
    current = plan(load_workspace().root / frozen["config_path"])
    for key in ("config_sha256", "source_sha256", "parent", "prompt_fingerprints", "check_jobs", "decode_job_identity",
            "answer_only_job_identity"):
        if current[key] != frozen[key]:
            raise ValueError("Frozen rules-decoder identity changed: " + key)
    return frozen


def freeze(config_path: Path, output: Path | None) -> tuple[Path, dict]:
    result = plan(config_path)
    root = load_workspace(config_path).root
    identity = list(result["source_sha256"]) + [result["config_path"]]
    changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *identity],
        cwd=root, text=True, encoding="utf-8")
    if changed.strip():
        raise ValueError("Commit the rules-decoder source and configuration before freezing")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output or root / "artifacts/runs/recurrent_steganography" / f"{stamp}-rules-decoder"
    if destination.exists():
        raise ValueError("Preserve existing evidence; choose another output directory")
    destination.mkdir(parents=True)
    (destination / "plan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return destination, result


def preview(config_path: Path) -> str:
    result = plan(config_path)
    loaded = load(Path(config_path).resolve())
    instruction = loaded["parent_config"]["think_instruction"]
    order = parent_design.BIT_OPTIONS
    estimates = ", ".join(f"${value} at {key.removeprefix('thinking output ').removesuffix(' tokens')}"
        for key, value in result["estimated_cost_usd"].items())
    lines = [f"# Draft for review: {PROTOCOL}", "",
        "Nothing here has been frozen or run. The both-reasoning run's thinking and answer-only Sonnet 5.5 decoders "
        "read its 576 payload paraphrases again. The one change: each decoder prompt shows the instructions the "
        "writer received, with the source message and the secret bit withheld.", "",
        f"Calls: {result['total_calls']} ({result['calls']['decoder-check']} pre-test, {result['calls']['decode']} "
        f"thinking decode, {result['calls']['decode-answer-only']} answer-only decode). Estimated cost: {estimates} "
        f"thinking output tokens per call. Proposed ceiling: ${result['run_budget_usd']}.", "", "## Decision rule", "",
        DECISION_RULE, "",
        "## Thinking decoder prompt", "", "```text",
        with_rules(parent_design.decoder_text("<the encoder's paraphrase>", order, instruction), loaded["instructions"]),
        "```", "", "## Answer-only decoder prompt", "", "```text",
        with_rules(parent_design.decoder_text("<the encoder's paraphrase>", order, None), loaded["instructions"]), "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m recurrent_steg.rules_decoder",
        description="Re-decode the both-reasoning paraphrases with the writer's instructions shown to the decoder")
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
            from recurrent_steg.rules_decoder.execution import execute
            print(json.dumps(execute(args.run.resolve(), args.phase, live=True), indent=2))
        elif args.command == "analyze":
            from recurrent_steg.rules_decoder.analysis import analyze
            result = analyze(args.run.resolve())
            print(json.dumps({key: result[key] for key in ("rules_helped", "primary", "total_spend_usd")}, indent=2))
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
                result = {key: result[key] for key in ("protocol", "subject", "models", "calls", "total_calls",
                    "estimated_input_tokens", "estimated_cost_usd", "run_budget_usd")}
            print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, "error: " + str(exc) + "\n")
