from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Keep these imports available for existing experiment callers.
from eval_core.models import (
    BudgetSettings, GenerationRequest, GenerationResponse, ModelConfig, RateCard, TokenUsage,
)

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
