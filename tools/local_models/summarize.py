"""Make a compact, reviewable summary from the full local acceptance artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.input = args.input.resolve()
    summary = {"schema_version": 1, "purpose": "Local runtime acceptance; not a steganography result", "models": {}}
    selected = {name: settings for name, settings in profiles().items()
        if (args.input / f"{name}-d{settings['depths'][0]}.json").exists()}
    if not selected:
        raise ValueError("No complete acceptance profile was found")
    for name, settings in selected.items():
        rows = []
        calibration = []
        for depth in settings["depths"]:
            path = args.input / f"{name}-d{depth}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data["passed"]:
                raise ValueError(f"Acceptance did not pass: {path}")
            benchmark = next(r["response"] for r in data["records"] if r["label"] == "benchmark")
            rows.append(dict(depth=depth, actual_iterations_verified=True,
                repeatable_after_intervening_request=data["repeatable_after_intervening_request"],
                cache_control_argmax_equal=all(x["argmax_equal"] for x in data["cache_control"]),
                cache_control_positions=len(data["cache_control"]),
                benchmark_output_tokens=benchmark["output_tokens"],
                benchmark_decode_tokens_per_second=benchmark["decode_tokens_per_second"],
                benchmark_wall_seconds=benchmark["elapsed_seconds"],
                largest_test_peak_allocated_gib=max(r["response"]["peak_allocated_bytes"] for r in data["records"]) / 1024**3,
                artifact=str(path.relative_to(ROOT)).replace("\\", "/"),
                artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for record in data["records"]:
                if "expected" in record:
                    calibration.append(dict(label=record["label"], expected=record["expected"],
                        text=record["response"]["text"], exact_match=record["exact_match"]))
        reference_path = args.input / f"{name}-upstream.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if not all(r["identical"] for r in reference["records"]):
            raise ValueError("Upstream comparison failed")
        summary["models"][name] = dict(model=data["runtime"]["model"],
            gpu=data["runtime"]["gpu"], versions=data["runtime"]["versions"],
            source_sha256=data["runtime"]["source_sha256"], adapter_sha256=data["runtime"]["adapter_sha256"],
            depth_checks=rows, calibration=calibration,
            upstream_token_identical_checks=len(reference["records"]),
            upstream_artifact=str(reference_path.relative_to(ROOT)).replace("\\", "/"),
            upstream_artifact_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            completed_at=data["completed_at"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, data in summary["models"].items():
        for row in data["depth_checks"]:
            print(f"{name} depth {row['depth']}: {row['benchmark_decode_tokens_per_second']:.1f} decode tokens/s; peak {row['largest_test_peak_allocated_gib']:.2f} GiB")
        print(f"{name} exact calibration: {sum(r['exact_match'] for r in data['calibration'])}/{len(data['calibration'])}")


if __name__ == "__main__":
    main()
