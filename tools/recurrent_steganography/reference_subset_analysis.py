"""Separately labeled subset analysis of a frozen paid-reference run.

The frozen analysis (`recurrent_steg.reference.analysis.analyze`) needs every
subject's phases. When a subject's pipeline was stopped, this script runs that same
frozen code restricted to the completed subjects, then applies the pre-specified
Holm adjustment over all configured subject-condition cells, counting every cell of
an excluded subject at p = 1. It writes `analysis-subset-<subjects>-v1.json` and
leaves no `analysis-v1.json` behind.

    .venv/Scripts/python.exe -B tools/recurrent_steganography/reference_subset_analysis.py \
        --run RUN_DIRECTORY --subject gpt55 --reason "why other subjects are excluded"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.workspace import load_workspace  # noqa: E402
from recurrent_steg.reference import analysis as frozen_analysis  # noqa: E402
from recurrent_steg.reference.cli import verify_freeze  # noqa: E402
from recurrent_steg.reference.design import CONDITIONS, load_design, subject_roles  # noqa: E402
from recurrent_steg.reference.scoring import holm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--subject", action="append", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    run_dir = args.run.resolve()
    plan = verify_freeze(run_dir)
    config, _, _, _ = load_design(load_workspace().root / plan["config_path"])
    roles = subject_roles(config)
    unknown = set(args.subject) - set(roles)
    if unknown:
        raise SystemExit(f"Unknown subjects: {sorted(unknown)}")
    selected = {subject: roles[subject] for subject in roles if subject in args.subject}
    destination = run_dir / f"analysis-subset-{'-'.join(selected)}-v1.json"
    full = run_dir / "analysis-v1.json"
    if destination.exists() or full.exists():
        raise SystemExit("Preserve earlier analysis; use an explicit versioned reanalysis")
    with patch.object(frozen_analysis, "subject_roles", return_value=selected):
        result = frozen_analysis.analyze(run_dir)
    full.rename(destination)  # the frozen function always names its output analysis-v1.json
    pvalues = {key: 1.0 if cell["primary_p"] is None else cell["primary_p"] for key, cell in result["cells"].items()}
    pvalues.update({f"{subject}/{condition}": 1.0 for subject in roles if subject not in selected
        for condition in CONDITIONS})
    adjusted = holm(pvalues)
    alpha = float(config["alpha"])
    for key, cell in result["cells"].items():
        cell["holm_p_within_subset"] = cell.pop("holm_p")
        cell["feasible_within_subset"] = cell.pop("feasible")
        cell["holm_p"] = adjusted[key]
        cell["feasible"] = bool(cell["primary_mean"] is not None and cell["primary_mean"] > 0
            and adjusted[key] < alpha)
    result["feasible_cells"] = sorted(key for key, cell in result["cells"].items() if cell["feasible"])
    result["analysis_role"] = "separately labeled subset analysis of a frozen exploratory reference run"
    result["subset"] = dict(subjects=list(selected), excluded=[s for s in roles if s not in selected],
        holm_cells=len(pvalues), excluded_cells_p=1.0, reason=args.reason,
        method="Frozen analyze() with the subject list restricted; Holm over all configured cells.")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(destination), feasible_cells=result["feasible_cells"],
        cells={key: dict(primary_mean=cell["primary_mean"], primary_p=cell["primary_p"], holm_p=cell["holm_p"])
            for key, cell in result["cells"].items()}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
