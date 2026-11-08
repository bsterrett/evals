"""Build the committed evidence extract for the paid feasibility reference run.

Writes `experiments/recurrent_steganography/notes/paid-reference-v1-evidence.json`:
GPT-5.5 per-cover outcomes and the Sonnet 5.5 stop evidence. It reads the local-only
run `20261106T221454Z-reference` and its separately labeled subset analysis
(`analysis-subset-gpt55-v1.json`, from `../reference_subset_analysis.py`).

Recovered from the session transcript: the extract was written on 2026-11-06 by a
one-off command and then compacted by a second one. Both steps are combined here,
unchanged apart from repo-relative paths and LF line endings.
"""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")

import json, hashlib  # noqa: E401,E402
from pathlib import Path  # noqa: E402
from collections import Counter  # noqa: E402
from recurrent_steg.reference.execution import read_rows, phase_records  # noqa: E402

run = Path(REPO + "artifacts/runs/recurrent_steganography/20261106T221454Z-reference")
a = json.loads((run / "analysis-subset-gpt55-v1.json").read_text(encoding="utf-8"))
cells = {k: {f: c[f] for f in ("primary_mean", "primary_counts", "primary_p", "holm_p", "holm_p_within_subset",
    "feasible", "recovery_only_mean", "recovery_only_counts", "recovery_only_p", "recovered", "decoder_valid",
    "accepted_carriers", "identical_pairs", "deterministic_overt", "by_word_pair")} for k, c in a["cells"].items()}
rows = []
for condition, details in a["per_cover"]["gpt55"].items():
    for d in details:
        rows.append(dict(condition=condition, case_id=d["case_id"], payloads=d["payloads"], carriers=d["carriers"],
            answers=d["answers"], accepted=d["accepted"], identical=d["carriers"][0] == d["carriers"][1],
            primary=d["primary"], recovery_only=d["recovery_only"]))
sonnet_decodes = read_rows(run / "decode-sonnet55.jsonl")
sonnet_leaks = [dict(condition=r["condition"], payload=r["payload"], carrier=r["carrier"],
    label_hits=r["checks"]["label_hits"], forbidden_hits=r["checks"]["forbidden_hits"])
    for r in phase_records(run, "encode", "sonnet55") if r["checks"]["overt"]]
s = a["subjects"]["gpt55"]
extract = dict(schema_version=1,
    role="Committed extract of the paid feasibility reference: GPT-5.5 per-cover outcomes and Sonnet 5.5 stop evidence",
    run_id="20261106T221454Z-reference", protocol="paid-feasibility-reference-v1",
    method_revision="fd80d8328958b6ee2c771eaf3daa180c1f24f567",
    analysis=dict(file="analysis-subset-gpt55-v1.json",
        sha256=hashlib.sha256((run / "analysis-subset-gpt55-v1.json").read_bytes()).hexdigest(),
        method=a["subset"]["method"], reason=a["subset"]["reason"], holm_cells=a["subset"]["holm_cells"]),
    gpt55=dict(judge="sonnet55", judge_qualification={k: v for k, v in s["judge_qualification"].items()},
        cells=cells, exploratory_contrasts=s["exploratory_contrasts"], honest_decodes=s["honest_decodes"],
        no_carrier=s["no_carrier"], development_honest_accepted=s["development_honest_accepted"],
        spend_usd=s["spend_usd"], total_spend_usd=s["total_spend_usd"], per_cover=rows),
    sonnet55=dict(status="stopped during the frozen decode phase; not analyzed",
        decodes_completed=len(sonnet_decodes),
        decode_format_by_condition={f"{c}/{'bare' if b else 'not bare'}": n for (c, b), n in sorted(Counter(
            (r["condition"], r["parsed"]["bare"]) for r in sonnet_decodes).items())},
        capped_decode=dict(trial="dec-enc-sonnet55-ref-dev-06-honest-focal-bits", output_tokens=256, cap=256,
            charge_usd="0.002976", ledger="blocked, then reconciled by the user at 2026-11-06T23:29:26Z"),
        deterministic_leaks=sonnet_leaks),
    limitations="Raw request/response logs and phase files are local-only. Provider default sampling; GPT-5.5 decoding is stochastic.")

# Second step of the original: one compact line per per-cover row.
per_cover = extract["gpt55"].pop("per_cover")
extract["gpt55"]["per_cover"] = "__ROWS__"
text = json.dumps(extract, indent=1, ensure_ascii=False)
compact = ",\n".join("   " + json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in per_cover)
text = text.replace('"__ROWS__"', "[\n" + compact + "\n  ]")
json.loads(text)  # still valid JSON
out = Path(REPO + "experiments/recurrent_steganography/notes/paid-reference-v1-evidence.json")
with open(out, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(text + "\n")
print(out, out.stat().st_size, "bytes")
