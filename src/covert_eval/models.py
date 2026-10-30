from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RateCard:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    max_input_tokens: int
    verified_at: date
    source: str
    cached_input_usd_per_million: Decimal | None = None
    cache_write_usd_per_million: Decimal | None = None

    def highest_input_rate(self) -> Decimal:
        return max(
            rate
            for rate in (
                self.input_usd_per_million,
                self.cached_input_usd_per_million,
                self.cache_write_usd_per_million,
            )
            if rate is not None
        )

    def reserve_cost(self, input_tokens: int, max_output_tokens: int) -> Decimal:
        if input_tokens < 0 or max_output_tokens < 1:
            raise ValueError("Token counts must be nonnegative and output must be positive")
        return (
            Decimal(input_tokens) * self.highest_input_rate()
            + Decimal(max_output_tokens) * self.output_usd_per_million
        ) / Decimal(1_000_000)

    def actual_cost(self, usage: "TokenUsage") -> Decimal:
        cached_rate = self.cached_input_usd_per_million or self.input_usd_per_million
        write_rate = self.cache_write_usd_per_million or self.input_usd_per_million
        return (
            Decimal(usage.input_tokens) * self.input_usd_per_million
            + Decimal(usage.cached_input_tokens) * cached_rate
            + Decimal(usage.cache_write_input_tokens) * write_rate
            + Decimal(usage.output_tokens) * self.output_usd_per_million
        ) / Decimal(1_000_000)


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    model: str
    rate: RateCard
    enabled: bool = True
    reasoning_effort: str | None = None
    thinking_type: str | None = None


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    hypothesis: str
    dataset: Path
    output_dir: Path
    seed: int
    repetitions: int
    max_cases: int
    cover_max_tokens: int
    decoder_max_tokens: int
    judge_max_tokens: int
    excluded_case_ids: tuple[str, ...] = ()
    condition_registry: Path | None = None
    case_split_manifest: Path | None = None
    default_condition_id: str | None = None
    default_analysis_role: str = "exploratory"


@dataclass(frozen=True)
class BudgetSettings:
    campaign_usd: Decimal
    ledger: Path
    pricing_max_age_days: int


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    source_path: Path
    experiment: ExperimentSettings
    budget: BudgetSettings
    subjects: tuple[ModelConfig, ...]
    judge: ModelConfig


@dataclass(frozen=True)
class EvalCase:
    id: str
    explicit_category: str
    implicit_category: str
    explicit_message: str
    implicit_message: str
    payload_source_id: str | None = None


@dataclass(frozen=True)
class GenerationRequest:
    stage: str
    prompt: str
    max_output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens + self.cache_write_input_tokens


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model: str
    usage: TokenUsage | None
    response_id: str | None
    status: str
    stop_reason: str | None
    raw: dict[str, Any]
    operational_error: str | None = None

