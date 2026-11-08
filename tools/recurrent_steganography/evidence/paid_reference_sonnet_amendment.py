"""Build the committed evidence extract for the answer-only Sonnet 5.5 amendment.

Writes `experiments/recurrent_steganography/notes/paid-reference-v1.1-sonnet-evidence.json`:
per-cover outcomes and the combined decision. It reads the local-only parent run
`20261106T221454Z-reference` and the amendment run
`20261107T003223Z-reference-sonnet-answer-only` with its `analysis-sonnet55-amended-v1.json`.

Recovered from the session transcript: the extract was written on 2026-11-07 (UTC)
by a one-off command. Unchanged apart from repo-relative paths and LF line endings.
"""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")

import json, hashlib  # noqa: E401,E402
from pathlib import Path  # noqa: E402
from recurrent_steg.reference.execution import phase_records  # noqa: E402

runs = Path(REPO + "artifacts/runs/recurrent_steganography")
run, amend = runs / "20261106T221454Z-reference", runs / "20261107T003223Z-reference-sonnet-answer-only"
a = json.loads((amend / "analysis-sonnet55-amended-v1.json").read_text(encoding="utf-8"))
enc = {r["id"]: r for r in phase_records(run, "encode", "sonnet55")}
judged = {r["source_id"]: r["judgment"] for r in phase_records(run, "judge", "sonnet55")}
idx = {(r["case_id"], r["condition"], r["payload"]): r for r in enc.values() if r["kind"] == "encoded"}
rows = []
for cond, details in a["per_cover"]["sonnet55"].items():
    for d in details:
        flags = []
        for p in d["payloads"]:
            r = idx[(d["case_id"], cond, p)]
            j = judged.get(r["id"])
            flags.append(dict(leak=r["checks"]["overt"], facts=j["facts_preserved"] if j else None,
                added=j["added_claims"] if j else None, suspicion=j["suspicion"] if j else None))
        rows.append(dict(condition=cond, case_id=d["case_id"], payloads=d["payloads"], carriers=d["carriers"],
            answers=d["answers"], accepted=d["accepted"], identical=d["carriers"][0] == d["carriers"][1],
            judge=flags, primary=d["primary"], recovery_only=d["recovery_only"]))
s = a["subjects"]["sonnet55"]
cells = {k: {f: c[f] for f in ("primary_mean", "primary_counts", "primary_p", "holm_p", "feasible",
    "recovery_only_mean", "recovery_only_counts", "recovery_only_p", "recovered", "decoder_valid",
    "accepted_carriers", "identical_pairs", "deterministic_overt", "by_word_pair")} for k, c in a["cells"].items()}
data = dict(schema_version=1,
    role="Committed extract of the answer-only Sonnet 5.5 amendment: per-cover outcomes and the combined decision",
    run_id=amend.name, protocol=a["amendment"]["protocol"], parent_run=run.name,
    method_revision="22d31d1", analysis=dict(file="analysis-sonnet55-amended-v1.json",
        sha256=hashlib.sha256((amend / "analysis-sonnet55-amended-v1.json").read_bytes()).hexdigest()),
    judge="gpt55", judge_qualification=s["judge_qualification"], cells=cells, combined_decision=a["combined_decision"],
    feasible_cells=a["feasible_cells"], exploratory_contrasts=s["exploratory_contrasts"],
    honest_decodes=s["honest_decodes"], no_carrier=s["no_carrier"],
    development_honest_accepted=s["development_honest_accepted"], per_cover="__ROWS__",
    limitations="Raw logs are local-only. Provider default sampling. Judge flags are GPT-5.5 judgments, not human ratings.")
text = json.dumps(data, indent=1, ensure_ascii=False)
text = text.replace('"__ROWS__"', "[\n" + ",\n".join("  " + json.dumps(r, ensure_ascii=False, separators=(",", ":"))
    for r in rows) + "\n ]")
json.loads(text)
out = Path(REPO + "experiments/recurrent_steganography/notes/paid-reference-v1.1-sonnet-evidence.json")
with open(out, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(text + "\n")
print(out, out.stat().st_size)
