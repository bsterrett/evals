from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (Path, date, datetime)):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, default=json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_fingerprint(root: Path) -> tuple[str, list[str]]:
    paths = [root / "pyproject.toml", root / "uv.lock"]
    paths.extend(sorted((root / "src").rglob("*.py")))
    present = [path for path in paths if path.is_file()]
    digest = hashlib.sha256()
    relative_names: list[str] = []
    for path in sorted(present, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        relative_names.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative_names


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_artifact_checksums(run_dir: Path) -> None:
    checksum_path = run_dir / "artifact_checksums.json"
    entries = {
        path.relative_to(run_dir).as_posix(): file_sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != checksum_path
    }
    write_json(
        checksum_path,
        {
            "algorithm": "sha256",
            "generated_at": datetime.now(timezone.utc),
            "files": entries,
        },
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:80]
