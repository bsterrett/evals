from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from covert_eval.models import EvalCase


SPLIT_SCHEMA_VERSION = 1


class SplitManifestError(ValueError):
    pass


@dataclass(frozen=True)
class CaseSplitManifest:
    schema_version: int
    manifest_id: str
    dataset_version: str
    source_path: Path
    development_case_ids: tuple[str, ...]
    confirmation_case_ids: tuple[str, ...]
    sha256: str
    frozen_at: str
    exposure_note: str

    def ids_for_role(self, analysis_role: str) -> tuple[str, ...]:
        if analysis_role == "exploratory":
            return self.development_case_ids
        if analysis_role == "confirmatory":
            return self.confirmation_case_ids
        raise SplitManifestError(f"Unknown analysis role {analysis_role!r}")


def _nonempty(value: Any, context: str) -> str:
    text = str(value).strip()
    if not text:
        raise SplitManifestError(f"{context} must be nonempty")
    return text


def load_split_manifest(path: str | Path, cases: list[EvalCase]) -> CaseSplitManifest:
    source_path = Path(path).resolve()
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise SplitManifestError(
            f"Unsupported split schema version {raw.get('schema_version') if isinstance(raw, dict) else None!r}"
        )
    partitions = raw.get("partitions")
    if not isinstance(partitions, dict):
        raise SplitManifestError("partitions must be an object")

    def case_ids(name: str) -> tuple[str, ...]:
        values = partitions.get(name)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise SplitManifestError(f"partitions.{name} must be a nonempty string list")
        normalized = tuple(value.strip() for value in values)
        if len(normalized) != len(set(normalized)):
            raise SplitManifestError(f"partitions.{name} contains duplicates")
        return normalized

    development = case_ids("development")
    confirmation = case_ids("confirmation")
    if set(development) & set(confirmation):
        raise SplitManifestError("Development and confirmation partitions overlap")
    known = {case.id for case in cases}
    partitioned = set(development) | set(confirmation)
    if partitioned != known:
        raise SplitManifestError(
            f"Split must cover the dataset exactly; missing={sorted(known - partitioned)}; "
            f"unknown={sorted(partitioned - known)}"
        )
    by_id = {case.id: case for case in cases}
    for name, ids in (("development", development), ("confirmation", confirmation)):
        id_set = set(ids)
        crossing = sorted(
            case.id for case in (by_id[value] for value in ids) if case.payload_source_id not in id_set
        )
        if crossing:
            raise SplitManifestError(
                f"partitions.{name} has payload_source_id links crossing the split: {crossing}"
            )
    for field in ("explicit_message", "implicit_message"):
        dev_text = {getattr(by_id[value], field).casefold() for value in development}
        test_text = {getattr(by_id[value], field).casefold() for value in confirmation}
        if dev_text & test_text:
            raise SplitManifestError(f"Duplicate {field} text crosses the split")

    return CaseSplitManifest(
        schema_version=SPLIT_SCHEMA_VERSION,
        manifest_id=_nonempty(raw.get("manifest_id", ""), "manifest_id"),
        dataset_version=_nonempty(raw.get("dataset_version", ""), "dataset_version"),
        source_path=source_path,
        development_case_ids=development,
        confirmation_case_ids=confirmation,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        frozen_at=_nonempty(raw.get("frozen_at", ""), "frozen_at"),
        exposure_note=_nonempty(raw.get("exposure_note", ""), "exposure_note"),
    )
