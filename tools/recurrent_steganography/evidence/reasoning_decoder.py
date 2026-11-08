"""Build the committed evidence extract for the reasoning-decoder amendment run."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import hashlib
import json
import re
from collections import Counter

RUN = REPO + "artifacts/runs/recurrent_steganography/20261107T055920Z-reasoning-decoder-v1.1/"
PARENT = REPO + "artifacts/runs/recurrent_steganography/20261107T022508Z-confirmation/"
OUT = REPO + "experiments/recurrent_steganography/notes/reasoning-decoder-v1.1-evidence.json"
MIN_REASONED = 40


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def words(text):
    return len(text.split())


def nsent(text):
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def counts(rows, key):
    out = Counter(key(row) for row in rows)
    return {"/".join(map(str, k)): v for k, v in sorted(out.items(), key=str)}


a = json.load(open(RUN + "analysis-v1.json", encoding="utf-8"))
decodes = load(RUN + "decode-sonnet55.jsonl")
checks = load(RUN + "decoder-check-sonnet55.jsonl")
carriers = {row["id"]: row for row in load(PARENT + "encode-sonnet55.jsonl")}
encoded = [row for row in decodes if row["kind"] == "encoded"]
for row in encoded:
    row["carrier"] = carriers[row["source_id"]]["carrier"]
    row["output_tokens"] = int((row.get("usage") or {}).get("output_tokens") or 0)
    row["reasoned"] = row["output_tokens"] >= MIN_REASONED
by_case = {}
for row in encoded:
    by_case.setdefault(row["case_id"], {})[row["target"]] = row

per_cover = []
for cover in a["per_cover"]:
    pair = [by_case[cover["case_id"]][payload] for payload in ("0", "1")]
    per_cover.append(dict(cover, output_tokens=[row["output_tokens"] for row in pair],
        reasoned=[row["reasoned"] for row in pair]))

differing_both = [c for c in per_cover if not c["identical"] and all(c["reasoned"])]
parity_rule = sum((words(r["carrier"]) % 2 == 1) == (r["target"] == "1") for r in encoded)
post_hoc = dict(
    status="Exploratory; chosen after seeing the result. Not part of the decision rule.",
    answers_by_payload=a["answers_by_payload"], answers_by_shown_order=a["answers_by_shown_order"],
    decoder_answer_by_word_count_parity=counts(encoded,
        lambda r: ("odd" if words(r["carrier"]) % 2 else "even", "answer", r["parsed"]["answer"])),
    decodes_matching_parity_rule=dict(all=sum((words(r["carrier"]) % 2 == 1) == (r["parsed"]["answer"] == "1")
        for r in encoded), of=len(encoded),
        reasoned=sum((words(r["carrier"]) % 2 == 1) == (r["parsed"]["answer"] == "1") for r in encoded if r["reasoned"]),
        of_reasoned=sum(r["reasoned"] for r in encoded)),
    encoder_payload_by_word_count_parity=counts(encoded,
        lambda r: ("payload", r["target"], "odd" if words(r["carrier"]) % 2 else "even")),
    parity_rule_recovers=dict(correct=parity_rule, of=len(encoded)),
    decoder_answer_by_sentence_count=counts(encoded, lambda r: ("sentences", nsent(r["carrier"]), "answer",
        r["parsed"]["answer"])),
    correct_by_reasoned=counts(encoded, lambda r: ("reasoned" if r["reasoned"] else "unreasoned",
        "correct" if r["parsed"]["answer"] == r["target"] else "wrong")),
    differing_pairs_both_reasoned_recovery={"+1": sum(c["recovery_only"] == 1 for c in differing_both),
        "-1": sum(c["recovery_only"] == -1 for c in differing_both), "pairs": len(differing_both)},
    reasoned_summaries_mentioning_odd_or_even=sum(bool(re.search(r"\bodd\b|\beven\b", r["thinking"]["summary"].lower()))
        for r in encoded if r["reasoned"]),
    pre_test=[dict(id=r["id"], answer=r["parsed"]["answer"], output_tokens=r["usage"]["output_tokens"],
        thinking_summary=r["thinking"]["summary"]) for r in checks],
    example_decode_summaries=[dict(id=r["id"], target=r["target"], answer=r["parsed"]["answer"],
        output_tokens=r["output_tokens"], thinking_summary=r["thinking"]["summary"])
        for r in sorted((r for r in encoded if r["reasoned"]), key=lambda r: r["id"])[:12]],
)

files = ["plan.json", "decoder-check-sonnet55.jsonl", "decode-sonnet55.jsonl", "analysis-v1.json"]
extract = dict(schema_version=1,
    role="Committed extract of the reasoning-decoder amendment: decision, per-cover outcomes, and post hoc tallies",
    run_id="20261107T055920Z-reasoning-decoder-v1.1", protocol=a["protocol"], method_revision="9b76761",
    decoder_instruction=a["decoder_instruction"], parent_run="20261107T022508Z-confirmation",
    stopped_v1_run=dict(run_id="20261107T054540Z-reasoning-decoder", protocol="sonnet-reasoning-decoder-v1",
        method_revision="73aeb85", pre_test="8/8 valid answers, 0/8 reasoned (0 thinking tokens each)", spend_usd="0.006662"),
    raw_evidence="local-only under artifacts/runs/recurrent_steganography/20261107T055920Z-reasoning-decoder-v1.1/",
    file_sha256={name: sha(RUN + name) for name in files},
    decision_rule=a["decision_rule"], signal=a["signal"], primary=a["primary"], secondary=a["secondary"],
    tallies=a["tallies"], reasoning=a["reasoning"], no_carrier=a["no_carrier"], refusals=a["refusals"],
    spend_usd=a["spend_usd"], total_spend_usd=a["total_spend_usd"],
    per_cover_fields=dict(answers="reasoning decoder's answers for the [payload 0, payload 1] carriers",
        answer_only_answers="the confirmation's answer-only answers", reasoned=f"output of at least {MIN_REASONED} tokens",
        carriers="the paraphrases themselves are in sonnet-confirmation-v1-evidence.json"),
    post_hoc=post_hoc,
    per_cover=[{k: v for k, v in c.items() if k != "family" or True} for c in per_cover])

with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(extract, handle, indent=1, ensure_ascii=False)
    handle.write("\n")
print(json.dumps({k: v for k, v in post_hoc.items() if k not in ("pre_test", "example_decode_summaries")}, indent=1))
print("summary sample:", post_hoc["example_decode_summaries"][0]["thinking_summary"][:500])
