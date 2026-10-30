from __future__ import annotations

import hashlib
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator

from eval_core.artifacts import append_jsonl, safe_name
from eval_core.budget import BudgetExceeded, BudgetLedger, Reservation
from eval_core.models import BudgetSettings, GenerationRequest, GenerationResponse, ModelConfig
from eval_core.providers import Provider, ProviderError, ProviderRefusal, create_provider


class RunSession:
    """Execution and cost accounting with no assumptions about an experiment's task.

    Calls within a session are sequential; independent sessions/processes may run
    concurrently against the same SQLite campaign ledger.
    """

    def __init__(
        self, *, experiment_id: str, output_dir: Path, budget: BudgetSettings,
        dry_run: bool = True, run_budget_usd: Decimal | None = None,
        run_name: str | None = None, provider_factory: Callable[..., Provider] = create_provider,
    ) -> None:
        if not experiment_id.strip():
            raise ValueError("An experiment id is required")
        if run_budget_usd is not None and (not run_budget_usd.is_finite() or run_budget_usd <= 0):
            raise ValueError("Run budget must be positive and finite")
        self.experiment_id = experiment_id
        self.budget_settings = budget
        self.dry_run = dry_run
        self.run_budget_usd = run_budget_usd
        self.run_spent_usd = Decimal()
        self.providers: dict[str, Provider] = {}
        self.provider_factory = provider_factory
        self.ledger: BudgetLedger | None = None
        self._call_lock = threading.Lock()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        mode = "dry" if dry_run else "live"
        self.run_id = f"{timestamp}-{safe_name(run_name or experiment_id)}-{mode}-{uuid.uuid4().hex[:8]}"
        self.run_dir = output_dir / self.run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.results_path = self.run_dir / "results.jsonl"

    @contextmanager
    def accounting(self) -> Iterator[RunSession]:
        # SQLite transactions cover ledger updates, never a provider request.
        self.ledger = BudgetLedger(
            self.budget_settings.ledger, self.budget_settings.campaign_usd,
            persist=not self.dry_run,
        )
        yield self

    def provider(self, model: ModelConfig) -> Provider:
        key = f"{model.provider}:{model.id}"
        if key not in self.providers:
            self.providers[key] = self.provider_factory(model, dry_run=self.dry_run)
        return self.providers[key]

    def budget(self) -> BudgetLedger:
        if self.ledger is None:
            raise RuntimeError("Budget ledger has not been initialized; use session.accounting()")
        return self.ledger

    def call(
        self, model: ModelConfig, request: GenerationRequest, *, trial_id: str,
    ) -> tuple[GenerationResponse, Decimal]:
        with self._call_lock:
            return self._execute_call(model, request, trial_id=trial_id)

    def _execute_call(
        self,
        model: ModelConfig,
        request: GenerationRequest,
        *,
        trial_id: str,
    ) -> tuple[GenerationResponse, Decimal]:
        if not self.dry_run:
            age = (date.today() - model.rate.verified_at).days
            if not 0 <= age <= self.budget_settings.pricing_max_age_days:
                raise ValueError(f"Pricing for {model.id} must be re-verified before live calls")
        provider = self.provider(model)
        input_tokens = provider.count_input_tokens(request)
        if input_tokens > model.rate.max_input_tokens:
            raise ProviderError(
                f"{model.id} {request.stage} input has {input_tokens} tokens, over configured limit "
                f"{model.rate.max_input_tokens}"
            )
        reserved_cost = model.rate.reserve_cost(input_tokens, request.max_output_tokens)
        call_meta = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "trial_id": trial_id,
            "stage": request.stage,
            "provider": model.provider,
            "model_id": model.id,
            "model": model.model,
            "reasoning_effort": model.reasoning_effort,
            "thinking_type": model.thinking_type,
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        }
        reservation: Reservation | None = None
        if not self.dry_run:
            if (
                self.run_budget_usd is not None
                and self.run_spent_usd + reserved_cost > self.run_budget_usd
            ):
                raise BudgetExceeded(
                    f"Next request needs up to ${reserved_cost:.6f}; settled run spend "
                    f"${self.run_spent_usd:.6f} plus that reservation would exceed the "
                    f"${self.run_budget_usd:.6f} run ceiling"
                )
            reservation = self.budget().reserve(reserved_cost, call_meta)
        append_jsonl(
            self.events_path,
            {
                "event": "call_started",
                "at": datetime.now(timezone.utc),
                **call_meta,
                "input_tokens_counted": input_tokens,
                "max_output_tokens": request.max_output_tokens,
                "reserved_usd": reserved_cost if not self.dry_run else Decimal(),
                "simulated_reservation_usd": reserved_cost if self.dry_run else None,
                "prompt": request.prompt,
                "request_metadata": request.metadata,
            },
        )
        try:
            response = provider.generate(request)
        except BaseException as exc:
            if reservation is not None:
                self.budget().mark_uncertain(reservation, call_meta, reason=repr(exc))
            append_jsonl(
                self.events_path,
                {
                    "event": "call_failed",
                    "at": datetime.now(timezone.utc),
                    **call_meta,
                    "error": repr(exc),
                    "budget_status": "uncertain" if reservation is not None else "dry_run",
                },
            )
            raise

        append_jsonl(
            self.events_path,
            {
                "event": "call_received",
                "at": datetime.now(timezone.utc),
                **call_meta,
                "response_id": response.response_id,
                "requested_model": model.model,
                "returned_model": response.model,
                "status": response.status,
                "stop_reason": response.stop_reason,
                "usage": asdict(response.usage) if response.usage is not None else None,
                "text": response.text,
                "operational_error": response.operational_error,
                "raw_response": response.raw,
            },
        )

        if response.usage is None:
            reason = response.operational_error or "Provider response did not include valid usage"
            if reservation is not None:
                self.budget().mark_uncertain(reservation, call_meta, reason=reason)
            append_jsonl(
                self.events_path,
                {
                    "event": "call_rejected",
                    "at": datetime.now(timezone.utc),
                    **call_meta,
                    "response_id": response.response_id,
                    "reason": reason,
                    "budget_status": "blocked_uncertain" if reservation is not None else "dry_run",
                },
            )
            raise ProviderError(reason)

        actual_cost = model.rate.actual_cost(response.usage)
        if response.model != model.model:
            reason = (
                f"Requested fixed model {model.model!r}, but provider reported {response.model!r}"
            )
            if reservation is not None:
                self.budget().mark_uncertain(
                    reservation,
                    call_meta,
                    reason=reason,
                    amount=max(reservation.amount, actual_cost),
                )
            append_jsonl(
                self.events_path,
                {
                    "event": "call_rejected",
                    "at": datetime.now(timezone.utc),
                    **call_meta,
                    "response_id": response.response_id,
                    "reason": reason,
                    "budget_status": "blocked_uncertain" if reservation is not None else "dry_run",
                },
            )
            raise ProviderError(reason)

        def reject_known_charge(reason: str) -> None:
            if reservation is not None:
                self.budget().settle_and_block(reservation, actual_cost, call_meta, reason)
            append_jsonl(
                self.events_path,
                {
                    "event": "call_rejected",
                    "at": datetime.now(timezone.utc),
                    **call_meta,
                    "response_id": response.response_id,
                    "reason": reason,
                    "usage": asdict(response.usage),
                    "actual_usd": actual_cost if not self.dry_run else Decimal(),
                    "simulated_actual_usd": actual_cost if self.dry_run else None,
                    "budget_status": "settled_blocked" if reservation is not None else "dry_run",
                },
            )
            raise ProviderError(reason)

        if response.usage.total_input_tokens > model.rate.max_input_tokens:
            reject_known_charge("Reported input usage exceeded the configured request bound")
        if response.stop_reason == "refusal":
            reason = response.operational_error or "Provider returned a refusal"
            stop_details = (
                response.raw.get("stop_details")
                if isinstance(response.raw, dict)
                else None
            )
            refusal_category = (
                str(stop_details.get("category"))
                if isinstance(stop_details, dict) and stop_details.get("category") is not None
                else None
            )
            if reservation is not None:
                self.budget().settle(
                    reservation,
                    actual_cost,
                    {**call_meta, "outcome": "provider_refusal"},
                )
                self.run_spent_usd += actual_cost
            append_jsonl(
                self.events_path,
                {
                    "event": "call_rejected",
                    "at": datetime.now(timezone.utc),
                    **call_meta,
                    "response_id": response.response_id,
                    "reason": reason,
                    "usage": asdict(response.usage),
                    "actual_usd": actual_cost if not self.dry_run else Decimal(),
                    "simulated_actual_usd": actual_cost if self.dry_run else None,
                    "budget_status": "settled_refusal" if reservation is not None else "dry_run",
                },
            )
            raise ProviderRefusal(
                reason,
                category=refusal_category,
                stop_details=stop_details,
            )
        if response.operational_error is not None or response.status != "completed":
            reject_known_charge(
                response.operational_error
                or f"Provider response ended with non-completed status {response.status!r}"
            )
        if reservation is not None:
            try:
                self.budget().settle(reservation, actual_cost, call_meta)
            except BaseException as exc:
                append_jsonl(
                    self.events_path,
                    {
                        "event": "call_rejected",
                        "at": datetime.now(timezone.utc),
                        **call_meta,
                        "response_id": response.response_id,
                        "reason": repr(exc),
                        "usage": asdict(response.usage),
                        "actual_usd": actual_cost,
                        "budget_status": "blocked",
                    },
                )
                raise
            self.run_spent_usd += actual_cost
        append_jsonl(
            self.events_path,
            {
                "event": "call_completed",
                "at": datetime.now(timezone.utc),
                **call_meta,
                "response_id": response.response_id,
                "requested_model": model.model,
                "returned_model": response.model,
                "status": response.status,
                "stop_reason": response.stop_reason,
                "usage": asdict(response.usage),
                "actual_usd": actual_cost if not self.dry_run else Decimal(),
                "simulated_actual_usd": actual_cost if self.dry_run else None,
                "text": response.text,
            },
        )
        return response, actual_cost
