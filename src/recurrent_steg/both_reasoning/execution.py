"""Phase runner for the both-reasoning variant: ordered phases, two pre-test gates, one spending ceiling."""
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
from recurrent_steg.both_reasoning.design import (PROTOCOL, SPLIT_SIZES, control_jobs, decoder_check_jobs,
    decoding_jobs, encoder_check_jobs, encoding_jobs, judge_jobs, load_design)
from recurrent_steg.both_reasoning.provider import ThinkingAnthropicProvider, thinking_tokens
from recurrent_steg.reasoning_decoder.provider import ReasoningAnswerAnthropicProvider, thinking_summary
from recurrent_steg.reference.execution import control_gate, phase_path, phase_records, read_rows
from recurrent_steg.reference.scoring import carrier_checks, parse_judgment
from recurrent_steg.reference_amendment.provider import AnswerOnlyAnthropicProvider, parse_answer

# Each phase needs the one before it. The primary measure is complete after "judge", so
# the answer-only secondary spends last.
PHASES = ("judge-controls", "encoder-check", "encode", "decoder-check", "decode", "judge", "decode-answer-only")
ENCODING = ("encoder-check", "encode")
REASONING_DECODE = ("decoder-check", "decode")
JUDGING = ("judge-controls", "judge")
# Input-token bound used only to size one maximal reservation in the cost projections.
RESERVATION_INPUT_TOKENS = 500


def live_factory(phase: str) -> Callable:
    if phase in ENCODING:
        return lambda model, dry_run=False: ThinkingAnthropicProvider(model)
    if phase in REASONING_DECODE:
        return lambda model, dry_run=False: ReasoningAnswerAnthropicProvider(model)
    if phase == "decode-answer-only":
        return lambda model, dry_run=False: AnswerOnlyAnthropicProvider(model)
    return create_provider


def spend(run_dir: Path, subject: str) -> Decimal:
    return sum((Decimal(row["cost_usd"]) for phase in PHASES
        for row in read_rows(phase_path(run_dir, phase, subject))), Decimal())


def reasoned(row: dict, config: dict) -> bool:
    """Counted from the thinking tokens the API reports, not from output length."""
    tokens = row.get("thinking_tokens")
    return type(tokens) is int and tokens >= config["reasoned_min_thinking_tokens"]


def _mean_cost(rows: list[dict]) -> Decimal:
    return sum((Decimal(row["cost_usd"]) for row in rows), Decimal()) / len(rows) if rows else Decimal()


def later_phases_usd(config: dict, decodes: int, judgments: int, *, reasoning_decodes: bool) -> Decimal:
    """Projected cost of the later phases from per-call costs measured in earlier runs, times the margin."""
    total = decodes * Decimal(config["answer_only_unit_usd"]) + judgments * Decimal(config["judge_unit_usd"])
    if reasoning_decodes:
        total += (config["decoder_check_jobs"] + decodes) * Decimal(config["decode_unit_usd"])
    return total * Decimal(config["projection_margin"])


def _gate(passed: bool, rows: list[dict], projected: Decimal, remaining: Decimal, **counts) -> dict:
    cent = Decimal("0.01")
    return dict(passed=passed, total=len(rows), **counts, mean_cost_usd=str(_mean_cost(rows).quantize(Decimal("0.000001"))),
        projected_usd=str(projected.quantize(cent)), remaining_usd=str(remaining.quantize(cent)))


def encoder_gate(rows: list[dict], config: dict, remaining: Decimal, pending: list[dict], decodes: int,
                 judgments: int, reservation: Decimal) -> dict:
    """Before the main encode: usable paraphrases, payload encodes that thought, and the rest of the run in budget.

    Payload and honest encodes are projected from their own pre-test means.
    """
    usable = sum(bool(row.get("carrier")) and row["checks"]["usable"] and not row["checks"]["overt"] for row in rows)
    payload_rows = [row for row in rows if row["kind"] == "encoded"]
    honest_rows = [row for row in rows if row["kind"] == "honest"]
    reasoned_count = sum(reasoned(row, config) for row in payload_rows)
    pending_payload = sum(job["kind"] == "encoded" for job in pending)
    encode_usd = _mean_cost(payload_rows) * pending_payload + _mean_cost(honest_rows) * (len(pending) - pending_payload)
    projected = (encode_usd * Decimal(config["projection_margin"]) + reservation
        + later_phases_usd(config, decodes, judgments, reasoning_decodes=True))
    passed = (len(rows) == 3 * SPLIT_SIZES["encoder_check"] and usable >= config["encoder_min_usable_checks"]
        and reasoned_count >= config["encoder_min_reasoned_checks"] and projected <= remaining)
    return _gate(passed, payload_rows, projected, remaining, check_calls=len(rows), usable=usable,
        reasoned=reasoned_count, honest_mean_cost_usd=str(_mean_cost(honest_rows).quantize(Decimal("0.000001"))),
        pending_jobs=len(pending))


def decoder_gate(rows: list[dict], config: dict, remaining: Decimal, pending: int, decodes: int, judgments: int,
                 reservation: Decimal) -> dict:
    """Before the main decode: valid answers, enough reasoning, and the remaining run inside the ceiling."""
    valid = sum(row["parsed"]["answer"] is not None for row in rows)
    reasoned_count = sum(reasoned(row, config) for row in rows)
    projected = (_mean_cost(rows) * pending * Decimal(config["projection_margin"]) + reservation
        + later_phases_usd(config, decodes, judgments, reasoning_decodes=False))
    passed = (len(rows) == config["decoder_check_jobs"] and valid == len(rows)
        and reasoned_count >= config["decoder_min_reasoned_checks"] and projected <= remaining)
    return _gate(passed, rows, projected, remaining, valid=valid, reasoned=reasoned_count, pending_jobs=pending)


def interpret(phase: str, job: dict, text: str | None, raw: dict | None) -> dict:
    if phase in JUDGING:
        try:
            return dict(judgment=parse_judgment(text), parse_error=None)
        except (ValueError, TypeError) as exc:
            return dict(judgment=None, parse_error=str(exc))
    if phase in ENCODING:
        carrier = text.strip() if text else None
        result = dict(carrier=carrier, checks=carrier_checks(carrier))
    else:
        result = dict(parsed=parse_answer(text, job["shown"]))
    if phase in ENCODING + REASONING_DECODE:
        result.update(thinking=thinking_summary(raw), thinking_tokens=thinking_tokens(raw))
    return result


def phase_jobs(phase: str, run_dir: Path, config: dict, cases: list[dict], models: dict) -> tuple[list[dict], object, int]:
    subject, judge = config["subject"], config["judge"]
    if phase == "judge-controls":
        return control_jobs(config, cases), models[judge], config["judge_tokens"]
    if phase == "encoder-check":
        return encoder_check_jobs(config, cases), models[subject], config["encoder_tokens"]
    if phase == "encode":
        return encoding_jobs(config, cases), models[subject], config["encoder_tokens"]
    carriers = phase_records(run_dir, "encode", subject)
    if phase == "decoder-check":
        return decoder_check_jobs(config, carriers, cases), models[subject], config["decoder_tokens"]
    if phase == "decode":
        return decoding_jobs(config, carriers, cases), models[subject], config["decoder_tokens"]
    if phase == "decode-answer-only":
        return (decoding_jobs(config, carriers, cases, reasoning=False), models[config["answer_only"]],
            config["answer_only_tokens"])
    return judge_jobs(config, carriers), models[judge], config["judge_tokens"]


def execute(run_dir: Path, phase: str, *, live: bool, provider_factory: Callable | None = None,
            echo: Callable[[str], None] = print) -> dict:
    from recurrent_steg.both_reasoning.cli import verify_freeze
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}")
    run_dir = Path(run_dir)
    frozen = verify_freeze(run_dir)
    workspace = load_workspace()
    config, cases, models, _ = load_design(workspace.root / frozen["config_path"])
    subject = config["subject"]
    path = phase_path(run_dir, phase, subject)
    receipt_path = path.with_suffix(".complete.json")
    if receipt_path.exists():
        return dict(already_complete=True, phase=phase, subject=subject, count=len(phase_records(run_dir, phase, subject)))
    position = PHASES.index(phase)
    if position:
        phase_records(run_dir, PHASES[position - 1], subject)  # raises unless the previous phase is complete
    if phase == "encoder-check" and not control_gate(phase_records(run_dir, "judge-controls", subject), config)["passed"]:
        raise ValueError("The judge failed its control check; no carriers will be generated")
    jobs, model, cap = phase_jobs(phase, run_dir, config, cases, models)
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
        reservation = models[subject].rate.reserve_cost(RESERVATION_INPUT_TOKENS,
            max(config["encoder_tokens"], config["decoder_tokens"]))
        if phase == "encode":
            gate = encoder_gate(phase_records(run_dir, "encoder-check", subject), config, remaining,
                [job for job in jobs if job["id"] not in done], 2 * SPLIT_SIZES["development"] + 2, len(jobs),
                reservation)
            if not gate["passed"]:
                raise ValueError("The reasoning encoder failed its pre-test; no carriers will be generated: "
                    + json.dumps(gate))
        elif phase == "decode":
            carriers = phase_records(run_dir, "encode", subject)
            gate = decoder_gate(phase_records(run_dir, "decoder-check", subject), config, remaining,
                len(jobs) - len(done), len(jobs), len(judge_jobs(config, carriers)), reservation)
            if not gate["passed"]:
                raise ValueError("The reasoning decoder failed its pre-test; no full decode will run: "
                    + json.dumps(gate))
        session = RunSession(experiment_id="recurrent_steganography", output_dir=run_dir / "calls",
            budget=workspace.budget, dry_run=not live, run_budget_usd=remaining,
            run_name=f"both-reasoning-{phase}", provider_factory=provider_factory or live_factory(phase))
        with session.accounting():
            for job in jobs:
                if job["id"] in done:
                    continue
                metadata = {"reference": dict(protocol=PROTOCOL, job_id=job["id"])}
                if phase not in JUDGING + ENCODING:
                    metadata["answer_choices"] = list(job["shown"])
                request = GenerationRequest(phase, job["prompt"], cap, metadata)
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
                row.update(meta, model_id=model.id, text=text, refusal=refusal,
                    cost_usd=str(session.run_spent_usd - before),
                    events=str(session.events_path.relative_to(run_dir)))
                row.update(interpret(phase, job, text, raw))
                append_jsonl(path, row)
                done.add(job["id"])
                echo(json.dumps(dict(phase=phase, done=len(done), total=len(jobs))))
        receipt = dict(phase=phase, subject=subject, count=len(done), sha256=file_sha256(path),
            completed_at=datetime.now(timezone.utc).isoformat(), live=live, gate=gate)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    finally:
        lock.unlink()
