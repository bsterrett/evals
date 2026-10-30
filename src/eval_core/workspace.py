from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from eval_core.models import BudgetSettings


def find_project_root(start: str | Path) -> Path:
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise ValueError(f"Cannot find the eval workspace above {start}")


@dataclass(frozen=True)
class Workspace:
    root: Path
    budget: BudgetSettings


def load_workspace(start: str | Path = ".") -> Workspace:
    root = find_project_root(start)
    with (root / "workspace.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported workspace schema")
    budget = raw["budget"]
    maximum = Decimal(budget["campaign_usd"])
    if not maximum.is_finite() or maximum <= 0:
        raise ValueError("Workspace campaign budget must be positive and finite")
    return Workspace(root, BudgetSettings(
        campaign_usd=maximum,
        ledger=(root / budget["ledger"]).resolve(),
        pricing_max_age_days=int(budget["pricing_max_age_days"]),
    ))


def git_provenance(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    try:
        return {
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
