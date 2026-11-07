"""Pre-test, decode, and answer-only phases for the rules decoder, with one gate and one spending ceiling."""
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
from eval_core.providers import ProviderRefusal
from eval_core.runtime import RunSession
from eval_core.workspace import load_workspace
from recurrent_steg.both_reasoning.provider import thinking_tokens
from recurrent_steg.reasoning_decoder.provider import ReasoningAnswerAnthropicProvider, thinking_summary
from recurrent_steg.reference.execution import phase_path, phase_records, read_rows
from recurrent_steg.reference_amendment.provider import AnswerOnlyAnthropicProvider, parse_answer
from recurrent_steg.rules_decoder.design import answer_only_jobs, check_jobs, decode_jobs, load

# Each phase needs the one before it. The primary is complete after "decode", so the
# answer-only secondary spends last.
PHASES = ("decoder-check", "decode", "decode-answer-only")
THINKING = ("decoder-check", "decode")
# Input-token bound used only to size one maximal reservation in the cost projection.
RESERVATION_INPUT_TOKENS = 700


def live_factory(phase: str) -> Callable:
    if phase in THINKING:
        return lambda model, dry_run=False: ReasoningAnswerAnthropicProvider(model)
    return lambda model, dry_run=False: AnswerOnlyAnthropicProvider(model)


def spend(run_dir: Path, subject: str) -> Decimal:
    return sum((Decimal(row["cost_usd"]) for phase in PHASES
        for row in read_rows(phase_path(run_dir, phase, subject))), Decimal())


def reasoned(row: dict, config: dict) -> bool:
    """Counted from the thinking tokens the API reports, not from output length."""
    tokens = row.get("thinking_tokens")
    return type(tokens) is int and tokens >= config["reasoned_min_thinking_tokens"]


def check_gate(rows: list[dict], config: dict, expected: int, remaining: Decimal, pending: int,
               answer_only: int, reservation: Decimal) -> dict:
    """All pre-test answers valid, enough reasoned, and the rest of the run projected inside the ceiling."""
    valid = sum(row["parsed"]["answer"] is not None for row in rows)
    reasoned_count = sum(reasoned(row, config) for row in rows)
    mean = sum((Decimal(row["cost_usd"]) for row in rows), Decimal()) / len(rows) if rows else Decimal()
    projected = ((mean * pending + Decimal(config["answer_only_unit_usd"]) * answer_only)
        * Decimal(config["projection_margin"]) + reservation)
    passed = (len(rows) == expected and valid == len(rows) and reasoned_count >= config["min_reasoned_checks"]
        and projected <= remaining)
    return dict(passed=passed, valid=valid, reasoned=reasoned_count, total=len(rows),
        mean_cost_usd=str(mean.quantize(Decimal("0.000001"))), pending_jobs=pending, answer_only_jobs=answer_only,
        projected_usd=str(projected.quantize(Decimal("0.01"))), remaining_usd=str(remaining.quantize(Decimal("0.01"))))


def phase_jobs(phase: str, loaded: dict) -> tuple[list[dict], object, int]:
    config = loaded["config"]
    if phase == "decoder-check":
        return check_jobs(loaded), loaded["model"], config["decoder_tokens"]
    if phase == "decode":
        return decode_jobs(loaded), loaded["model"], config["decoder_tokens"]
    return answer_only_jobs(loaded), loaded["answer_only_model"], config["answer_only_tokens"]


def execute(run_dir: Path, phase: str, *, live: bool, provider_factory: Callable | None = None,
            echo: Callable[[str], None] = print) -> dict:
    from recurrent_steg.rules_decoder.cli import verify_freeze
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}")
    run_dir = Path(run_dir)
    frozen = verify_freeze(run_dir)
    workspace = load_workspace()
    loaded = load(workspace.root / frozen["config_path"])
    config, subject = loaded["config"], loaded["subject"]
    path = phase_path(run_dir, phase, subject)
    receipt_path = path.with_suffix(".complete.json")
    if receipt_path.exists():
        return dict(already_complete=True, phase=phase, subject=subject, count=len(phase_records(run_dir, phase, subject)))
    position = PHASES.index(phase)
    if position:
        phase_records(run_dir, PHASES[position - 1], subject)  # raises unless the previous phase is complete
    jobs, model, cap = phase_jobs(phase, loaded)
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
        gate = None
        if phase == "decode":
            gate = check_gate(phase_records(run_dir, "decoder-check", subject), config, len(check_jobs(loaded)),
                remaining, len(jobs) - len(done), len(answer_only_jobs(loaded)),
                model.rate.reserve_cost(RESERVATION_INPUT_TOKENS, config["decoder_tokens"]))
            if not gate["passed"]:
                raise ValueError("The rules decoder failed its pre-test; no full decode will run: " + json.dumps(gate))
        session = RunSession(experiment_id="recurrent_steganography", output_dir=run_dir / "calls",
            budget=workspace.budget, dry_run=not live, run_budget_usd=remaining, run_name=f"rules-decoder-{phase}",
            provider_factory=provider_factory or live_factory(phase))
        with session.accounting():
            for job in jobs:
                if job["id"] in done:
                    continue
                request = GenerationRequest(phase, job["prompt"], cap, {"answer_choices": list(job["shown"]),
                    "reference": dict(protocol=config["protocol"], job_id=job["id"])})
                before = session.run_spent_usd
                try:
                    response, _ = session.call(model, request, trial_id=job["id"])
                    text, raw, refusal = response.text, response.raw, None
                    meta = dict(response_id=response.response_id, returned_model=response.model,
                        stop_reason=response.stop_reason,
                        usage=asdict(response.usage) if response.usage is not None else None)
                except ProviderRefusal as exc:
                    text, raw, refusal = None, None, str(exc)
                    meta = dict(response_id=None, returned_model=None, stop_reason="refusal", usage=None,
                        refusal_category=exc.category)
                row = {key: value for key, value in job.items() if key != "prompt"}
                row.update(meta, model_id=model.id, text=text, refusal=refusal, parsed=parse_answer(text, job["shown"]),
                    cost_usd=str(session.run_spent_usd - before), events=str(session.events_path.relative_to(run_dir)))
                if phase in THINKING:
                    row.update(thinking=thinking_summary(raw), thinking_tokens=thinking_tokens(raw))
                append_jsonl(path, row)
                done.add(job["id"])
                echo(json.dumps(dict(phase=phase, done=len(done), total=len(jobs))))
        receipt = dict(phase=phase, subject=subject, count=len(done), sha256=file_sha256(path),
            completed_at=datetime.now(timezone.utc).isoformat(), live=live, gate=gate)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    finally:
        lock.unlink()
