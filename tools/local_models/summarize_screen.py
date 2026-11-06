"""Summarize completed diagnostic screens without discarding censored attempts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    cohorts = {}
    originals = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "records" not in data:
            continue
        originals[path.stem] = data
        rows = []
        for record in data["records"]:
            response = record.get("response", {})
            if response and any(n != data["runtime"]["depth"] for n in response["observed_iterations"]):
                raise ValueError("An observed depth did not match the configured depth")
            rows.append(dict(case=record["case"]["id"], expected=record["case"]["expected"],
                final_text=record.get("final_text"), exact_correct=record.get("score", {}).get("exact_correct", False),
                output_tokens=response.get("output_tokens"), stop_reason=response.get("stop_reason"),
                output_limit_reached=record.get("output_limit_reached"), error=record.get("error")))
        cohorts[path.stem] = dict(profile=data["runtime"]["profile"], model=data["runtime"]["model"],
            precision=data["runtime"]["precision"], placement=data["runtime"].get("placement"),
            depth=data["runtime"]["depth"], latent_policy=data["runtime"].get("latent_policy"),
            prompt_format=data["prompt_format"], max_output_tokens=data["max_output_tokens"],
            operational_success=data["operational_success"], exact_correct=data["exact_correct"],
            cases_total=data["cases_total"],
            peak_allocated_gib=max(r["response"]["peak_allocated_bytes"] for r in data["records"] if "response" in r)/1024**3,
            observed_depth_verified=True, cases=rows,
            artifact=str(path.relative_to(ROOT)).replace("\\", "/"),
            artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    a = originals["huginn-bfloat16-native"]
    b = originals["huginn-float32-native"]
    if [r["request"] for r in a["records"]] != [r["request"] for r in b["records"]]:
        raise ValueError("Huginn precision comparisons used different requests")
    paired = dict(same_requests=True, same_model=a["runtime"]["model"] == b["runtime"]["model"],
        identical_output_token_sequences=sum(x["response"]["output_token_ids"] == y["response"]["output_token_ids"]
            for x,y in zip(a["records"],b["records"])), cases=len(a["records"]),
        exact_scores_equal=a["exact_correct"] == b["exact_correct"])
    short = next(r for r in originals["dory-d2"]["records"] if r["case"]["id"] == "decode-0")["response"]
    long = originals["dory-d2-extended-decode0"]["records"][0]["response"]
    followup = dict(initial_output_tokens=short["output_tokens"], extended_output_tokens=long["output_tokens"],
        prefix_identical=short["output_token_ids"] == long["output_token_ids"][:short["output_tokens"]])
    result = dict(schema_version=1, purpose="Capability and precision diagnostics; not a steganography experiment",
        scoring_note="Exact scores concern final-answer compliance. The raw single-distinct-symbol extractor is conservative and is not a semantic ability estimate.",
        cohorts=cohorts, huginn_paired_precision=paired, dory_censoring_followup=followup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, row in cohorts.items():
        print(f"{name}: {row['exact_correct']}/{row['cases_total']} exact; {row['peak_allocated_gib']:.2f} GiB peak")
    print("Huginn paired precision:", paired)
    print("Dory extended budget:", followup)


if __name__ == "__main__":
    main()
