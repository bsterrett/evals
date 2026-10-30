from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONDITION_SCHEMA_VERSION = 1
ANALYSIS_ROLES = frozenset({"exploratory", "confirmatory"})
FACTOR_KEYS = frozenset(
    {
        "carrier_style_instruction",
        "carrier_word_range",
        "explicit_retention",
        "additional_assertions",
        "payload_expression",
        "channel_family",
        "shared_protocol",
        "encoder_process",
        "decoder_awareness",
    }
)


class ConditionRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentalCondition:
    id: str
    prompt_release: str
    scoring_protocol_id: str
    analysis_plan_id: str
    claim_family: str
    strict_claim_eligible: bool
    confirmatory_eligible: bool
    parent_condition_id: str | None
    changed_from_parent: tuple[str, ...]
    factors: dict[str, str]

    def content_sha256(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConditionRegistry:
    schema_version: int
    registry_id: str
    source_path: Path
    conditions: dict[str, ExperimentalCondition]
    sha256: str

    def get(self, condition_id: str) -> ExperimentalCondition:
        try:
            return self.conditions[condition_id]
        except KeyError as exc:
            raise ConditionRegistryError(f"Unknown condition id {condition_id!r}") from exc


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty(value: Any, context: str) -> str:
    text = str(value).strip()
    if not text:
        raise ConditionRegistryError(f"{context} must be nonempty")
    return text


def load_condition_registry(path: str | Path) -> ConditionRegistry:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != CONDITION_SCHEMA_VERSION:
        raise ConditionRegistryError(
            f"Unsupported condition schema version {raw.get('schema_version')!r}"
        )
    registry_id = _nonempty(raw.get("registry_id", ""), "registry_id")
    raw_conditions = raw.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ConditionRegistryError("At least one [[conditions]] entry is required")

    conditions: dict[str, ExperimentalCondition] = {}
    for index, item in enumerate(raw_conditions):
        context = f"conditions[{index}]"
        if not isinstance(item, dict):
            raise ConditionRegistryError(f"{context} must be a table")
        condition_id = _nonempty(item.get("id", ""), f"{context}.id")
        if condition_id in conditions:
            raise ConditionRegistryError(f"Duplicate condition id {condition_id!r}")
        raw_factors = item.get("factors")
        if not isinstance(raw_factors, dict):
            raise ConditionRegistryError(f"{context}.factors must be a table")
        factor_keys = set(raw_factors)
        if factor_keys != FACTOR_KEYS:
            raise ConditionRegistryError(
                f"{context}.factors must contain exactly {sorted(FACTOR_KEYS)}; "
                f"missing={sorted(FACTOR_KEYS - factor_keys)}; "
                f"unexpected={sorted(factor_keys - FACTOR_KEYS)}"
            )
        factors = {
            key: _nonempty(raw_factors[key], f"{context}.factors.{key}")
            for key in sorted(FACTOR_KEYS)
        }
        changed_raw = item.get("changed_from_parent", [])
        if not isinstance(changed_raw, list) or not all(
            isinstance(value, str) and value.strip() for value in changed_raw
        ):
            raise ConditionRegistryError(
                f"{context}.changed_from_parent must be a list of factor paths"
            )
        changed = tuple(value.strip() for value in changed_raw)
        if len(changed) != len(set(changed)):
            raise ConditionRegistryError(f"{context}.changed_from_parent has duplicates")
        unknown_changed = set(changed) - FACTOR_KEYS
        if unknown_changed:
            raise ConditionRegistryError(
                f"{context}.changed_from_parent has unknown factors: {sorted(unknown_changed)}"
            )
        parent_value = item.get("parent_condition_id")
        parent = _nonempty(parent_value, f"{context}.parent_condition_id") if parent_value else None
        strict = item.get("strict_claim_eligible")
        if type(strict) is not bool:
            raise ConditionRegistryError(f"{context}.strict_claim_eligible must be boolean")
        confirmatory = item.get("confirmatory_eligible")
        if type(confirmatory) is not bool:
            raise ConditionRegistryError(f"{context}.confirmatory_eligible must be boolean")
        conditions[condition_id] = ExperimentalCondition(
            id=condition_id,
            prompt_release=_nonempty(item.get("prompt_release", ""), f"{context}.prompt_release"),
            scoring_protocol_id=_nonempty(
                item.get("scoring_protocol_id", ""), f"{context}.scoring_protocol_id"
            ),
            analysis_plan_id=_nonempty(
                item.get("analysis_plan_id", ""), f"{context}.analysis_plan_id"
            ),
            claim_family=_nonempty(item.get("claim_family", ""), f"{context}.claim_family"),
            strict_claim_eligible=strict,
            confirmatory_eligible=confirmatory,
            parent_condition_id=parent,
            changed_from_parent=changed,
            factors=factors,
        )

    for condition in conditions.values():
        if condition.parent_condition_id is None:
            if condition.changed_from_parent:
                raise ConditionRegistryError(
                    f"Root condition {condition.id!r} cannot declare changed_from_parent"
                )
            continue
        if condition.parent_condition_id not in conditions:
            raise ConditionRegistryError(
                f"Condition {condition.id!r} has unknown parent {condition.parent_condition_id!r}"
            )
        parent = conditions[condition.parent_condition_id]
        actual_changes = tuple(
            sorted(
                key
                for key in FACTOR_KEYS
                if condition.factors[key] != parent.factors[key]
            )
        )
        if tuple(sorted(condition.changed_from_parent)) != actual_changes:
            raise ConditionRegistryError(
                f"Condition {condition.id!r} changed_from_parent does not match factor diff; "
                f"expected {list(actual_changes)}"
            )

    for condition in conditions.values():
        seen: set[str] = set()
        current: ExperimentalCondition | None = condition
        while current is not None:
            if current.id in seen:
                raise ConditionRegistryError(f"Condition ancestry cycle at {current.id!r}")
            seen.add(current.id)
            current = (
                conditions[current.parent_condition_id]
                if current.parent_condition_id is not None
                else None
            )

    return ConditionRegistry(
        schema_version=CONDITION_SCHEMA_VERSION,
        registry_id=registry_id,
        source_path=source_path,
        conditions=conditions,
        sha256=_file_sha256(source_path),
    )


def legacy_v1_4_condition() -> ExperimentalCondition:
    return ExperimentalCondition(
        id="legacy_v1_4_strict_trace_standard",
        prompt_release="v1.4.0",
        scoring_protocol_id="cover-coupled-v1.0",
        analysis_plan_id="legacy-v1",
        claim_family="strict_zero_shot_reasoning_like",
        strict_claim_eligible=True,
        confirmatory_eligible=False,
        parent_condition_id=None,
        changed_from_parent=(),
        factors={
            "additional_assertions": "forbidden",
            "carrier_style_instruction": "reasoning_trace",
            "carrier_word_range": "60-140",
            "channel_family": "ordinary_linguistic_choices",
            "decoder_awareness": "explicit_forced_candidate",
            "encoder_process": "one_shot",
            "explicit_retention": "full",
            "payload_expression": "no_overt_expression",
            "shared_protocol": "none",
        },
    )
