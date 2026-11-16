"""Build the compact, tracked summary of a retrofit acceptance directory.

Reads the full records written by retrofit_check.py (ignored by Git) and keeps
the measurements needed to assess them, with each record's SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def design(entry: dict) -> dict:
    rows = list(entry.get("rows", {}).values())
    peaks = [row["peak_allocated_GiB"] for row in rows if row["peak_allocated_GiB"]]
    result = {
        "program": entry["program"], "passes": entry["passes"], "layer_calls": entry["layer_calls"],
        "update_cells": entry["update_cells"], "error": entry.get("error"),
        "calls_and_slots_verified_every_forward": entry.get("calls_and_slots_verified_every_forward", False),
    }
    if "error" in entry:
        return result
    result.update({
        "cache_gap": entry["cache_control"]["max_absolute_logit_difference"],
        "cache_argmax_equal": f"{entry['cache_control']['argmax_equal']}/{entry['cache_control']['steps']}",
        "explicit_gap": entry["explicit"]["max_absolute_logit_difference"],
        "first_step_gap_from_parent": entry["differs_from_parent"]["max_absolute_logit_difference"],
        "mean_decode_tokens_per_second": round(sum(row["decode_tokens_per_second"] for row in rows) / len(rows), 2),
        "peak_allocated_GiB": max(peaks) if peaks else None,
        "texts": {label: row["text"] for label, row in entry["rows"].items()},
    })
    if "stock_copy_cached" in entry:
        result["stock_copy_identical"] = {mode: entry[f"stock_copy_{mode}"]["identical"] for mode in ("cached", "uncached")}
    return result


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    runtime = data["runtime"]
    parent_rows = list(data["parent"].values())
    peaks = [row["peak_allocated_GiB"] for row in parent_rows if row["peak_allocated_GiB"]]
    return {
        "record": path.name, "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "passed": data["passed"], "failures": data["failures"], "completed_at": data["completed_at"],
        "parent": runtime["parent"], "parent_layers": runtime["parent_layers"],
        "loop_unit": data["unit"], "update_unit": data["update_unit"],
        "precision": data["precision"], "placement": data["placement"],
        "tokens_per_prompt": data["tokens_per_prompt"], "prompts": data["prompts"],
        "versions": runtime["versions"], "gpu": runtime["gpu"],
        "gaps_decided": data["precision"] == "float32",
        "tolerance": data["tolerances"]["fp32_max_absolute_logit_difference"],
        "one_pass_identical_to_parent": {entry["program"]: entry["identical"] for entry in data["parent_equivalents"]},
        "parent_restored_after_removal": data["restored_parent"]["identical"],
        "parent_cache_gap": data["parent_cache_control"]["max_absolute_logit_difference"],
        "parent_explicit_gap": data["parent_explicit"]["max_absolute_logit_difference"],
        "parent_mean_decode_tokens_per_second": round(sum(row["decode_tokens_per_second"] for row in parent_rows) / len(parent_rows), 2),
        "parent_peak_allocated_GiB": max(peaks) if peaks else None,
        "parent_texts": {label: row["text"] for label, row in data["parent"].items()},
        "loops": [design(entry) for entry in data["loops"]],
        "updates": [design(entry) for entry in data["updates"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", help="A caveat to store with the summary")
    args = parser.parse_args()
    records = sorted(args.input.resolve().glob("*.json"))
    if not records:
        raise ValueError("No acceptance records found")
    summary = {"schema_version": 2,
        "purpose": "Retrofit runner acceptance; validates the tool, not whether any design helps a model",
        "source_directory": args.input.as_posix(), "note": args.note,
        "records": [summarize(path) for path in records]}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summarized {len(records)} records; all passed: {all(r['passed'] for r in summary['records'])}")


if __name__ == "__main__":
    main()
