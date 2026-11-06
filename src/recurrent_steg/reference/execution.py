"""Phase runner for the paid reference check, with per-subject spending shares."""
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
from recurrent_steg.reference.design import (PROTOCOL, control_jobs, decoding_jobs, encoding_jobs, judge_jobs,
    load_design, subject_roles)
from recurrent_steg.reference.scoring import carrier_checks, parse_choice, parse_judgment, quality_ok

PHASES = ("judge-controls", "encode", "decode", "judge")


def phase_path(run_dir: Path, phase: str, subject: str) -> Path:
    return Path(run_dir) / f"{phase}-{subject}.jsonl"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def phase_records(run_dir: Path, phase: str, subject: str) -> list[dict]:
    path = phase_path(run_dir, phase, subject)
    receipt_path = path.with_suffix(".complete.json")
    if not receipt_path.exists():
        raise ValueError(f"Phase {phase} for {subject} is not complete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError("Completed phase records changed: " + str(path))
    rows = read_rows(path)
    if len(rows) != receipt["count"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Completed phase has missing or duplicate records")
    return rows


def subject_spend(run_dir: Path, subject: str) -> Decimal:
    """Settled spend for one subject's pipeline, including its judge's calls."""
    return sum((Decimal(row["cost_usd"]) for phase in PHASES
        for row in read_rows(phase_path(run_dir, phase, subject))), Decimal())


def control_gate(rows: list[dict], config: dict) -> dict:
    """Contract and quality checks on known controls, before any carrier is generated."""
    rates = {}
    for kind in ("quality_copy", "quality_corrupt"):
        subset = [row for row in rows if row["kind"] == kind]
        rates[kind] = sum(row["judgment"] is not None and quality_ok(row["judgment"]) == (kind == "quality_copy")
            for row in subset) / len(subset)
    overt = [row for row in rows if row["kind"] == "overt"]
    valid = sum(row["judgment"] is not None for row in rows)
    minimum = float(config["judge_min_quality_control_rate"])
    return dict(passed=valid == len(rows) and all(rate >= minimum for rate in rates.values()),
        valid=valid, total=len(rows), quality_rate_by_kind=rates,
        overt_flagged=sum(row["judgment"] is not None and row["judgment"]["overt_payload"] for row in overt),
        overt_total=len(overt))


def interpret(phase: str, job: dict, text: str | None) -> dict:
    if phase in ("judge-controls", "judge"):
        try:
            return dict(judgment=parse_judgment(text), parse_error=None)
        except (ValueError, TypeError) as exc:
            return dict(judgment=None, parse_error=str(exc))
    if phase == "encode":
        carrier = text.strip() if text else None
        words = job["condition"] is not None and job["condition"].endswith("words")
        return dict(carrier=carrier, checks=carrier_checks(carrier, tuple(job["word_pair"]) if words else None))
    return dict(parsed=parse_choice(text, job["options"]))


def execute(run_dir: Path, phase: str, subject: str, *, live: bool,
            provider_factory: Callable = create_provider, echo: Callable[[str], None] = print) -> dict:
    from recurrent_steg.reference.cli import verify_freeze
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}")
    run_dir = Path(run_dir)
    frozen = verify_freeze(run_dir)
    workspace = load_workspace()
    config, cases, models, _ = load_design(workspace.root / frozen["config_path"])
    roles = subject_roles(config)
    if subject not in roles:
        raise ValueError(f"Unknown subject {subject!r}")
    judge = roles[subject]
    path = phase_path(run_dir, phase, subject)
    receipt_path = path.with_suffix(".complete.json")
    if receipt_path.exists():
        return dict(already_complete=True, phase=phase, subject=subject,
            count=len(phase_records(run_dir, phase, subject)))
    if phase == "judge-controls":
        jobs, model, cap = control_jobs(config, cases, subject, judge), models[judge], config["judge_tokens"]
    elif phase == "encode":
        gate = control_gate(phase_records(run_dir, "judge-controls", subject), config)
        if not gate["passed"]:
            raise ValueError("The judge failed its control check; no carriers will be generated")
        jobs, model, cap = encoding_jobs(config, cases, subject), models[subject], config["encoder_tokens"]
    elif phase == "decode":
        carriers = phase_records(run_dir, "encode", subject)
        jobs, model, cap = decoding_jobs(config, carriers, cases, subject), models[subject], config["decoder_tokens"]
    else:
        carriers = phase_records(run_dir, "encode", subject)
        jobs, model, cap = judge_jobs(config, carriers, cases, subject, judge), models[judge], config["judge_tokens"]
    remaining = Decimal(config["run_budget_usd"]) / len(roles) - subject_spend(run_dir, subject)
    if remaining <= 0:
        raise ValueError(f"The spending share for {subject} is exhausted")
    lock = path.with_suffix(".lock")
    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    try:
        existing = read_rows(path)
        done = {row["id"] for row in existing}
        if len(done) != len(existing) or not done <= {job["id"] for job in jobs}:
            raise ValueError("Partial phase has unexpected records")
        session = RunSession(experiment_id="recurrent_steganography", output_dir=run_dir / "calls",
            budget=workspace.budget, dry_run=not live, run_budget_usd=remaining,
            run_name=f"reference-{phase}-{subject}", provider_factory=provider_factory)
        with session.accounting():
            for job in jobs:
                if job["id"] in done:
                    continue
                request = GenerationRequest(phase, job["prompt"], cap,
                    {"reference": dict(protocol=PROTOCOL, job_id=job["id"])})
                before = session.run_spent_usd
                try:
                    response, _ = session.call(model, request, trial_id=job["id"])
                    text, refusal = response.text, None
                    meta = dict(response_id=response.response_id, returned_model=response.model,
                        stop_reason=response.stop_reason,
                        usage=asdict(response.usage) if response.usage is not None else None)
                except ProviderRefusal as exc:
                    # Settled by the session; recorded as a behavioral outcome, not retried.
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
                echo(json.dumps(dict(phase=phase, subject=subject, done=len(done), total=len(jobs))))
        receipt = dict(phase=phase, subject=subject, count=len(done), sha256=file_sha256(path),
            completed_at=datetime.now(timezone.utc).isoformat(), live=live)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    finally:
        lock.unlink()
