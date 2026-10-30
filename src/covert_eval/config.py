from __future__ import annotations

import json
import tomllib
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from covert_eval.models import (
    AppConfig,
    BudgetSettings,
    EvalCase,
    ExperimentSettings,
    ModelConfig,
    RateCard,
)
from covert_eval.conditions import ANALYSIS_ROLES, load_condition_registry
from covert_eval.splits import load_split_manifest


class ConfigError(ValueError):
    pass


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing {context}.{key}")
    return mapping[key]


def _decimal(value: Any, context: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ConfigError(f"{context} must be a decimal number") from exc
    if result < 0:
        raise ConfigError(f"{context} must be nonnegative")
    return result


def _positive_int(value: Any, context: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ConfigError(f"{context} must be a {qualifier} integer")
    return value


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _rate(data: dict[str, Any], context: str) -> RateCard:
    try:
        verified_at = date.fromisoformat(str(_required(data, "verified_at", context)))
    except ValueError as exc:
        raise ConfigError(f"{context}.verified_at must use YYYY-MM-DD") from exc
    rate = RateCard(
        input_usd_per_million=_decimal(
            _required(data, "input_usd_per_million", context),
            f"{context}.input_usd_per_million",
        ),
        cached_input_usd_per_million=(
            _decimal(data["cached_input_usd_per_million"], f"{context}.cached_input_usd_per_million")
            if "cached_input_usd_per_million" in data
            else None
        ),
        cache_write_usd_per_million=(
            _decimal(data["cache_write_usd_per_million"], f"{context}.cache_write_usd_per_million")
            if "cache_write_usd_per_million" in data
            else None
        ),
        output_usd_per_million=_decimal(
            _required(data, "output_usd_per_million", context),
            f"{context}.output_usd_per_million",
        ),
        max_input_tokens=_positive_int(
            _required(data, "max_input_tokens", context), f"{context}.max_input_tokens"
        ),
        verified_at=verified_at,
        source=str(_required(data, "source", context)),
    )
    if rate.input_usd_per_million == 0 or rate.output_usd_per_million == 0:
        raise ConfigError(f"{context} input and output prices must be greater than zero")
    if not rate.source.startswith("https://"):
        raise ConfigError(f"{context}.source must be an HTTPS URL")
    return rate


def _model(data: dict[str, Any], context: str, *, enabled_default: bool) -> ModelConfig:
    provider = str(_required(data, "provider", context)).lower()
    if provider not in {"openai", "anthropic", "mock"}:
        raise ConfigError(f"{context}.provider must be openai, anthropic, or mock")
    return ModelConfig(
        id=str(_required(data, "id", context)),
        provider=provider,
        model=str(_required(data, "model", context)),
        enabled=bool(data.get("enabled", enabled_default)),
        reasoning_effort=(str(data["reasoning_effort"]) if "reasoning_effort" in data else None),
        thinking_type=(str(data["thinking_type"]) if "thinking_type" in data else None),
        rate=_rate(_required(data, "rate", context), f"{context}.rate"),
    )


def load_config(path: str | Path) -> AppConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    schema_version = _positive_int(_required(raw, "schema_version", "root"), "schema_version")
    if schema_version not in {1, 2}:
        raise ConfigError(f"Unsupported schema_version {schema_version}")

    exp = _required(raw, "experiment", "root")
    budget = _required(raw, "budget", "root")
    excluded_case_ids_raw = exp.get("excluded_case_ids", [])
    if not isinstance(excluded_case_ids_raw, list) or not all(
        isinstance(case_id, str) and case_id.strip() for case_id in excluded_case_ids_raw
    ):
        raise ConfigError("experiment.excluded_case_ids must be a list of nonempty strings")
    excluded_case_ids = tuple(case_id.strip() for case_id in excluded_case_ids_raw)
    if len(excluded_case_ids) != len(set(excluded_case_ids)):
        raise ConfigError("experiment.excluded_case_ids must not contain duplicates")
    experiment = ExperimentSettings(
        name=str(_required(exp, "name", "experiment")),
        hypothesis=str(_required(exp, "hypothesis", "experiment")),
        dataset=_resolve(source_path, str(_required(exp, "dataset", "experiment"))),
        output_dir=_resolve(source_path, str(_required(exp, "output_dir", "experiment"))),
        seed=_positive_int(_required(exp, "seed", "experiment"), "experiment.seed", allow_zero=True),
        repetitions=_positive_int(_required(exp, "repetitions", "experiment"), "experiment.repetitions"),
        max_cases=_positive_int(_required(exp, "max_cases", "experiment"), "experiment.max_cases"),
        cover_max_tokens=_positive_int(
            _required(exp, "cover_max_tokens", "experiment"), "experiment.cover_max_tokens"
        ),
        decoder_max_tokens=_positive_int(
            _required(exp, "decoder_max_tokens", "experiment"), "experiment.decoder_max_tokens"
        ),
        judge_max_tokens=_positive_int(
            _required(exp, "judge_max_tokens", "experiment"), "experiment.judge_max_tokens"
        ),
        excluded_case_ids=excluded_case_ids,
        condition_registry=(
            _resolve(source_path, str(exp["condition_registry"]))
            if "condition_registry" in exp
            else None
        ),
        case_split_manifest=(
            _resolve(source_path, str(exp["case_split_manifest"]))
            if "case_split_manifest" in exp
            else None
        ),
        default_condition_id=(
            str(exp["default_condition_id"]).strip()
            if "default_condition_id" in exp
            else None
        ),
        default_analysis_role=str(exp.get("default_analysis_role", "exploratory")).strip(),
    )
    budget_settings = BudgetSettings(
        campaign_usd=_decimal(_required(budget, "campaign_usd", "budget"), "budget.campaign_usd"),
        ledger=_resolve(source_path, str(_required(budget, "ledger", "budget"))),
        pricing_max_age_days=_positive_int(
            _required(budget, "pricing_max_age_days", "budget"), "budget.pricing_max_age_days"
        ),
    )
    if budget_settings.campaign_usd <= 0:
        raise ConfigError("budget.campaign_usd must be greater than zero")

    subjects_raw = _required(raw, "subjects", "root")
    if not isinstance(subjects_raw, list) or not subjects_raw:
        raise ConfigError("At least one [[subjects]] entry is required")
    subjects = tuple(
        _model(item, f"subjects[{index}]", enabled_default=True)
        for index, item in enumerate(subjects_raw)
    )
    judge = _model(_required(raw, "judge", "root"), "judge", enabled_default=True)
    config = AppConfig(
        schema_version=schema_version,
        source_path=source_path,
        experiment=experiment,
        budget=budget_settings,
        subjects=subjects,
        judge=judge,
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if not config.experiment.dataset.is_file():
        raise ConfigError(f"Dataset does not exist: {config.experiment.dataset}")
    ids = [subject.id for subject in config.subjects]
    if len(ids) != len(set(ids)):
        raise ConfigError("Subject ids must be unique")
    if not any(subject.enabled for subject in config.subjects):
        raise ConfigError("At least one subject must be enabled")
    for subject in config.subjects:
        if subject.provider == "openai" and subject.reasoning_effort != "none":
            raise ConfigError(f"V1 OpenAI subject {subject.id} must set reasoning_effort = 'none'")
        if subject.provider == "anthropic" and subject.thinking_type not in {"disabled", "between_tools"}:
            raise ConfigError(
                f"V1 Anthropic subject {subject.id} must disable thinking with "
                "thinking_type = 'disabled' or 'between_tools'"
            )
    cases = load_cases(config.experiment.dataset)
    case_ids = {case.id for case in cases}
    excluded = set(config.experiment.excluded_case_ids)
    unknown_exclusions = excluded - case_ids
    if unknown_exclusions:
        raise ConfigError(
            "experiment.excluded_case_ids contains unknown ids: "
            + ", ".join(sorted(unknown_exclusions))
        )
    if config.experiment.max_cases > len(cases) - len(excluded):
        raise ConfigError("experiment.max_cases exceeds the non-excluded dataset size")
    experiment = config.experiment
    condition_fields = (
        experiment.condition_registry,
        experiment.case_split_manifest,
        experiment.default_condition_id,
    )
    if config.schema_version == 1:
        if any(value is not None for value in condition_fields):
            raise ConfigError("schema_version 1 cannot declare condition or split fields")
    else:
        if any(value is None for value in condition_fields):
            raise ConfigError(
                "schema_version 2 requires experiment.condition_registry, "
                "case_split_manifest, and default_condition_id"
            )
        if experiment.default_analysis_role not in ANALYSIS_ROLES:
            raise ConfigError(
                f"experiment.default_analysis_role must be one of {sorted(ANALYSIS_ROLES)}"
            )
        try:
            registry = load_condition_registry(experiment.condition_registry)
            registry.get(str(experiment.default_condition_id))
            split = load_split_manifest(experiment.case_split_manifest, cases)
            from covert_eval.prompt_releases import get_prompt_release

            for condition in registry.conditions.values():
                get_prompt_release(condition)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        role_ids = split.ids_for_role(experiment.default_analysis_role)
        if experiment.max_cases > len(role_ids):
            raise ConfigError(
                "experiment.max_cases exceeds the default analysis-role partition size"
            )


def validate_live_config(config: AppConfig, *, today: date | None = None) -> None:
    today = today or date.today()
    models = [subject for subject in config.subjects if subject.enabled] + [config.judge]
    for model in models:
        age = (today - model.rate.verified_at).days
        if age < 0:
            raise ConfigError(f"Pricing for {model.id} is dated in the future")
        if age > config.budget.pricing_max_age_days:
            raise ConfigError(
                f"Pricing for {model.id} is {age} days old; re-verify it and update verified_at"
            )
        if "REPLACE" in model.model.upper():
            raise ConfigError(f"Replace the placeholder model id for {model.id}")


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                case = EvalCase(
                    id=str(item["id"]),
                    explicit_category=str(item["explicit_category"]),
                    implicit_category=str(item["implicit_category"]),
                    explicit_message=str(item["explicit_message"]).strip(),
                    implicit_message=str(item["implicit_message"]).strip(),
                    payload_source_id=str(item["payload_source_id"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ConfigError(f"Invalid dataset row at line {line_number}: {exc}") from exc
            if case.id in seen:
                raise ConfigError(f"Duplicate case id {case.id}")
            if not case.explicit_message or not case.implicit_message:
                raise ConfigError(f"Case {case.id} has an empty message")
            if not case.payload_source_id:
                raise ConfigError(f"Case {case.id} has an empty payload_source_id")
            if case.payload_source_id == case.id:
                raise ConfigError(
                    f"Case {case.id} reuses its original payload; V1 requires a deranged pairing"
                )
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ConfigError("Dataset is empty")
    payload_sources = [case.payload_source_id for case in cases]
    if len(payload_sources) != len(set(payload_sources)):
        raise ConfigError("payload_source_id values must be unique")
    if set(payload_sources) != {case.id for case in cases}:
        raise ConfigError("payload_source_id values must be a permutation of dataset case ids")
    return cases

