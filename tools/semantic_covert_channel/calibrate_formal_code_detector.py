"""Calibrate the formal-code detector on passages written under the ban.

Reads saved live results (encoded and control passages) and runs the detector
against each passage's own target and against every other development target.
None of these passages was allowed a formal code, so every flag is a false
positive. Makes no model calls.

    uv run python tools/semantic_covert_channel/calibrate_formal_code_detector.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from covert_eval.formal_codes import ATTEMPT_THRESHOLD, DETECTOR_VERSION, detect_formal_code


ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "experiments" / "semantic_covert_channel" / "data" / "v2_cases.jsonl"


def _passages() -> list[tuple[str, str, str, str]]:
    found: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    for results in sorted((ROOT / "artifacts" / "runs").rglob("results.jsonl")):
        if "-live-" not in results.parent.name:
            continue
        with results.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                target = row.get("implicit_message")
                for arm in ("encoded", "control"):
                    text = row.get(f"{arm}_cover")
                    if isinstance(text, str) and text.strip() and isinstance(target, str):
                        found[(arm, text)] = (results.parent.name, arm, text, target)
    return list(found.values())


def main() -> int:
    passages = _passages()
    dev_targets = []
    with CASES.open("r", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            if case["id"].startswith("v2-dev-"):
                dev_targets.append(case["implicit_message"])
    own_flags = 0
    own_best: list[float] = []
    cross_flags = 0
    cross_total = 0
    cross_best: list[float] = []
    flagged_schemes: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    for run, arm, text, target in passages:
        own = detect_formal_code(text, target)
        own_best.append(own["best_fraction"])
        if own["attempted"]:
            own_flags += 1
            flagged_schemes[own["best_scheme"]] += 1
            examples.append({"run": run, "arm": arm, "target": target, "kind": "own", **own})
        for other in dev_targets:
            if other == target:
                continue
            cross = detect_formal_code(text, other)
            cross_total += 1
            cross_best.append(cross["best_fraction"])
            if cross["attempted"]:
                cross_flags += 1
                flagged_schemes[cross["best_scheme"]] += 1
                if len(examples) < 12:
                    examples.append(
                        {"run": run, "arm": arm, "target": other, "kind": "cross", **cross}
                    )
    report = {
        "detector_version": DETECTOR_VERSION,
        "attempt_threshold": ATTEMPT_THRESHOLD,
        "distinct_passages": len(passages),
        "own_target": {
            "flagged": own_flags,
            "max_best_fraction": max(own_best) if own_best else None,
        },
        "other_targets": {
            "comparisons": cross_total,
            "flagged": cross_flags,
            "max_best_fraction": max(cross_best) if cross_best else None,
        },
        "flagged_by_scheme": dict(flagged_schemes),
        "examples": examples,
    }
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
