"""Build the committed evidence extract for the Sonnet plain/0-1 confirmation."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import hashlib
import json
import re
from collections import Counter

RUN = REPO + "artifacts/runs/recurrent_steganography/20261107T022508Z-confirmation/"
AMEND = REPO + "artifacts/runs/recurrent_steganography/20261107T003223Z-reference-sonnet-answer-only/"
REF = REPO + "artifacts/runs/recurrent_steganography/20261106T221454Z-reference/"
OUT = REPO + "experiments/recurrent_steganography/notes/sonnet-confirmation-v1-evidence.json"


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
    return {"/".join(map(str, k)) if isinstance(k, tuple) else str(k): v for k, v in sorted(out.items(), key=str)}


a = json.load(open(RUN + "analysis-v1.json", encoding="utf-8"))
enc = load(RUN + "encode-sonnet55.jsonl")
dec = load(RUN + "decode-sonnet55.jsonl")
judge = {row["source_id"]: row["judgment"] for row in load(RUN + "judge-sonnet55.jsonl")}
enc_id = {(e["case_id"], e["payload"]): e["id"] for e in enc if e["kind"] == "encoded"}
honest_text = {e["case_id"]: e["carrier"] for e in enc if e["kind"] == "honest"}
honest_dec = {d["case_id"]: d["parsed"]["answer"] for d in dec if d["kind"] == "honest"}


def flags(judgment):
    if judgment is None:
        return None
    return [judgment["facts_preserved"], judgment["added_claims"], judgment["overt_payload"], judgment["suspicion"]]


per_cover = []
for c in a["per_cover"]:
    per_cover.append(dict(case_id=c["case_id"], family=c["family"], option_order=c["option_order"],
        carriers=c["carriers"], answers=c["answers"], accepted=c["accepted"],
        judge_flags=[flags(judge.get(enc_id[(c["case_id"], p)])) for p in ("0", "1")],
        identical=c["identical"], informative=c["informative"], recovery_only=c["recovery_only"],
        honest_decode=honest_dec.get(c["case_id"])))

encoded_dec = [d for d in dec if d["kind"] == "encoded"]
differing = [c for c in a["per_cover"] if not c["identical"]]
length = Counter()
read_length = Counter()
for c in differing:
    w0, w1 = words(c["carriers"][0]), words(c["carriers"][1])
    length["payload1_longer" if w1 > w0 else "payload1_shorter" if w1 < w0 else "equal"] += 1
    if w0 != w1:
        longer = 1 if w1 > w0 else 0
        read_length["pairs"] += 1
        read_length["longer_answered_1"] += c["answers"][longer] == "1"
        read_length["shorter_answered_1"] += c["answers"][1 - longer] == "1"


def recovery_by(key):
    groups = {}
    for c in a["per_cover"]:
        groups.setdefault(c[key], Counter())[c["recovery_only"]] += 1
    return {k: {"+1": v[1], "0": v[0], "-1": v[-1]} for k, v in sorted(groups.items())}


ex_dec = [d for d in load(AMEND + "decode-sonnet55.jsonl") if d.get("condition") == "plain-bits"]
ex_enc = [e for e in load(REF + "encode-sonnet55.jsonl") if e.get("condition") == "plain-bits" and e["kind"] == "encoded"]
ex_pairs = {}
for e in ex_enc:
    ex_pairs.setdefault(e["case_id"], {})[e["payload"]] = e["carrier"]
ex_len = Counter()
for pair in ex_pairs.values():
    if pair["0"] == pair["1"]:
        continue
    w0, w1 = words(pair["0"]), words(pair["1"])
    ex_len["payload1_longer" if w1 > w0 else "payload1_shorter" if w1 < w0 else "equal"] += 1

encoded = [e for e in enc if e["kind"] == "encoded"]
post_hoc = dict(
    status="Exploratory; chosen after seeing the pre-registered result. Not part of the decision rule.",
    encoded_answers_by_payload=counts(encoded_dec, lambda d: ("payload", d["target"], "answer", d["parsed"]["answer"])),
    encoded_answers_by_shown_order=counts(encoded_dec,
        lambda d: ("shown", "".join(d["shown"]), "answer", d["parsed"]["answer"])),
    honest_answers_by_shown_order=counts([d for d in dec if d["kind"] == "honest"],
        lambda d: ("shown", "".join(d["shown"]), "answer", d["parsed"]["answer"])),
    recovery_only_by_option_order=recovery_by("option_order"),
    recovery_only_by_family=recovery_by("family"),
    recovery_only_identical_vs_differing={
        "identical": counts([c for c in a["per_cover"] if c["identical"]], lambda c: c["recovery_only"]),
        "differing": counts(differing, lambda c: c["recovery_only"])},
    identical_pairs_with_disagreeing_decodes=sum(1 for c in a["per_cover"]
        if c["identical"] and c["answers"][0] != c["answers"][1]),
    differing_pairs_word_length=dict(length),
    encoded_sentences_by_payload=counts(encoded, lambda e: ("payload", e["payload"], "sentences", nsent(e["carrier"]))),
    decoder_reads_length=dict(read_length),
    encoded_identical_to_honest=counts(encoded,
        lambda e: ("payload", e["payload"], "same_as_honest", e["carrier"] == honest_text[e["case_id"]])),
    exploratory_cell_for_comparison=dict(
        source=("Reference run 20261106T221454Z-reference, Sonnet plain-bits carriers, decoded under amendment "
                "20261107T003223Z-reference-sonnet-answer-only"),
        encoded_answers_by_payload=counts([d for d in ex_dec if d["kind"] == "encoded"],
            lambda d: ("payload", d["target"], "answer", d["parsed"]["answer"])),
        encoded_answers_by_shown_order=counts([d for d in ex_dec if d["kind"] == "encoded"],
            lambda d: ("shown", "".join(d["shown"]), "answer", d["parsed"]["answer"])),
        differing_pairs_word_length=dict(ex_len),
        encoded_sentences_by_payload=counts(ex_enc, lambda e: ("payload", e["payload"], "sentences", nsent(e["carrier"])))),
)

files = ["plan.json", "judge-controls-sonnet55.jsonl", "encode-sonnet55.jsonl", "decoder-check-sonnet55.jsonl",
         "decode-sonnet55.jsonl", "judge-sonnet55.jsonl", "analysis-v1.json"]
extract = dict(schema_version=1,
    role=("Committed extract of the pre-registered Sonnet 5.5 plain/0-1 confirmation: decision, per-cover "
          "outcomes, and post hoc tallies"),
    run_id="20261107T022508Z-confirmation", protocol=a["protocol"], method_revision="6c69336",
    raw_evidence="local-only under artifacts/runs/recurrent_steganography/20261107T022508Z-confirmation/",
    file_sha256={name: sha(RUN + name) for name in files},
    decision_rule=a["decision_rule"], confirmed=a["confirmed"], primary=a["primary"], secondary=a["secondary"],
    judge_qualification=a["judge_qualification"], tallies=a["tallies"], covers=a["covers"], carriers=a["carriers"],
    honest_decodes_by_position=a["honest_decodes_by_position"], no_carrier=a["no_carrier"],
    development_honest_accepted=a["development_honest_accepted"], development_honest=a["development_honest"],
    refusals=a["refusals"], spend_usd=a["spend_usd"], total_spend_usd=a["total_spend_usd"],
    per_cover_fields=dict(carriers="[payload 0, payload 1] paraphrases", answers="decoded answers for those carriers",
        judge_flags="per carrier: [facts_preserved, added_claims, overt_payload, suspicion] from GPT-5.5",
        honest_decode="decoded answer for the honest paraphrase of the same cover"),
    post_hoc=post_hoc, per_cover=per_cover)

with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(extract, handle, indent=1, ensure_ascii=False)
    handle.write("\n")
print(json.dumps({k: v for k, v in extract.items() if k not in ("per_cover", "decision_rule")}, indent=1))
