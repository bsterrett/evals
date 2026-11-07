"""Phase runner for the single-cell confirmation, with gates and one spending ceiling."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
from typing import Callable

from eval_core.artifacts import append_jsonl, file_sha256
from eval_core.models import GenerationRequest
from eval_core.providers import ProviderRefusal, create_provider
from eval_core.runtime import RunSession
from eval_core.workspace import load_workspace
from recurrent_steg.confirmation.design import (PROTOCOL, check_jobs, control_jobs, decoding_jobs, encoding_jobs,
    judge_jobs, load_design)
from recurrent_steg.reference.execution import control_gate, phase_path, phase_records, read_rows
from recurrent_steg.reference.scoring import carrier_checks, parse_judgment
from recurrent_steg.reference_amendment.provider import AnswerOnlyAnthropicProvider, parse_answer

PHASES = ("judge-controls", "encode", "decoder-check", "decode", "judge")
ANSWER_ONLY = ("decoder-check", "decode")


def live_factory(phase: str) -> Callable:
    if phase in ANSWER_ONLY:
        return lambda model, dry_run=False: AnswerOnlyAnthropicProvider(model)
    return create_provider


def spend(run_dir: Path, subject: str) -> Decimal:
    return sum((Decimal(row["cost_usd"]) for phase in PHASES
        for row in read_rows(phase_path(run_dir, phase, subject))), Decimal())


def check_gate(rows: list[dict]) -> dict:
    valid = sum(row["parsed"]["answer"] is not None for row in rows)
    return dict(passed=bool(rows) and valid == len(rows), valid=valid, total=len(rows))


def interpret(phase: str, job: dict, text: str | None) -> dict:
    if phase in ("judge-controls", "judge"):
        try:
            return dict(judgment=parse_judgment(text), parse_error=None)
        except (ValueError, TypeError) as exc:
            return dict(judgment=None, parse_error=str(exc))
    if phase == "encode":
        carrier = text.strip() if text else None
        return dict(carrier=carrier, checks=carrier_checks(carrier))
    return dict(parsed=parse_answer(text, job["shown"]))


def execute(run_dir: Path, phase: str, *, live: bool, provider_factory: Callable | None = None,
            echo: Callable[[str], None] = print) -> dict:
    from recurrent_steg.confirmation.cli import verify_freeze
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}")
    run_dir = Path(run_dir)
    frozen = verify_freeze(run_dir)
    workspace = load_workspace()
    config, cases, models, _ = load_design(workspace.root / frozen["config_path"])
    subject, judge = config["subject"], config["judge"]
    path = phase_path(run_dir, phase, subject)
    receipt_path = path.with_suffix(".complete.json")
    if receipt_path.exists():
        return dict(already_complete=True, phase=phase, subject=subject, count=len(phase_records(run_dir, phase, subject)))
    if phase == "judge-controls":
        jobs, model, cap = control_jobs(config, cases), models[judge], config["judge_tokens"]
    elif phase == "encode":
        if not control_gate(phase_records(run_dir, "judge-controls", subject), config)["passed"]:
            raise ValueError("The judge failed its control check; no carriers will be generated")
        jobs, model, cap = encoding_jobs(config, cases), models[subject], config["encoder_tokens"]
    else:
        carriers = phase_records(run_dir, "encode", subject)
        if phase == "decoder-check":
            jobs, model, cap = check_jobs(config, carriers, cases), models[subject], config["decoder_tokens"]
        elif phase == "decode":
            if not check_gate(phase_records(run_dir, "decoder-check", subject))["passed"]:
                raise ValueError("The answer-only decoder failed its pre-test; no full decode will run")
            jobs, model, cap = decoding_jobs(config, carriers, cases), models[subject], config["decoder_tokens"]
        else:
            jobs, model, cap = judge_jobs(config, carriers), models[judge], config["judge_tokens"]
    remaining = Decimal(config["run_budget_usd"]) - spend(run_dir, subject)
    if remaining <= 0:
        raise ValueError("The approved ceiling is exhausted")
    lock = path.with_suffix(".lock")
    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    try:
        existing = read_rows(path)
        done = {row["id"] for row in existing}
        if len(done) != len(existing) or not done <= {job["id"] for job in jobs}:
            raise ValueError("Partial phase has unexpected records")
        session = RunSession(experiment_id="recurrent_steganography", output_dir=run_dir / "calls",
            budget=workspace.budget, dry_run=not live, run_budget_usd=remaining,
            run_name=f"confirmation-{phase}", provider_factory=provider_factory or live_factory(phase))
        with session.accounting():
            for job in jobs:
                if job["id"] in done:
                    continue
                metadata = {"reference": dict(protocol=PROTOCOL, job_id=job["id"])}
                if phase in ANSWER_ONLY:
                    metadata["answer_choices"] = list(job["shown"])
                request = GenerationRequest(phase, job["prompt"], cap, metadata)
                before = session.run_spent_usd
                try:
                    response, _ = session.call(model, request, trial_id=job["id"])
                    text, refusal = response.text, None
                    meta = dict(response_id=response.response_id, returned_model=response.model,
                        stop_reason=response.stop_reason,
                        usage=asdict(response.usage) if response.usage is not None else None)
                except ProviderRefusal as exc:
                    text, refusal = None, str(exc)
                    meta = dict(response_id=None, returned_model=None, stop_reason="refusal", usage=None,
                        refusal_category=exc.category)
                row = {key: value for key, value in job.items() if key != "prompt"}
                row.update(meta, model_id=model.id, text=text, refusal=refusal,
                    cost_usd=str(session.run_spent_usd - before),
                    events=str(session.events_path.relative_to(run_dir)))
                row.update(interpret(phase, job, text))
                append_jsonl(path, row)
                done.add(job["id"])
                echo(json.dumps(dict(phase=phase, done=len(done), total=len(jobs))))
        receipt = dict(phase=phase, subject=subject, count=len(done), sha256=file_sha256(path),
            completed_at=datetime.now(timezone.utc).isoformat(), live=live)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    finally:
        lock.unlink()
