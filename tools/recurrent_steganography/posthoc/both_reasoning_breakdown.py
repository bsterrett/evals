"""Post hoc (exploratory) breakdown of the both-reasoning run. Chosen after seeing the pre-registered result."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import json
import re
import sys
from collections import Counter

from recurrent_steg.reasoning_decoder.analysis import exact_sign_flip_p  # noqa: E402

RUN = REPO + "artifacts/runs/recurrent_steganography/20261107T132019Z-both-reasoning/"
MIN_THINK = 20


def load(name):
    with open(RUN + name, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def words(text):
    return len(text.split())


a = json.load(open(RUN + "analysis-v1.json", encoding="utf-8"))
enc = load("encode-sonnet55.jsonl")
dec = load("decode-sonnet55.jsonl")
ans = load("decode-answer-only-sonnet55.jsonl")
carriers = {row["id"]: row for row in enc}
payload_rows = [row for row in enc if row["kind"] == "encoded"]


def sentences(row):
    return row["checks"]["sentences"]


def sentence_rule(row):
    return "1" if sentences(row) == 1 else "0"


def mentions(text, pattern):
    return bool(re.search(pattern, text.lower()))


out = {}
out["sentence_rule_recovers"] = dict(correct=sum(sentence_rule(r) == r["payload"] for r in payload_rows),
    of=len(payload_rows))
# Pair score of the sentence rule over development covers, with the run's own acceptance decisions.
scores = []
for cover in a["per_cover"]:
    rows = [next(r for r in payload_rows if r["case_id"] == cover["case_id"] and r["payload"] == p) for p in "01"]
    answers = [sentence_rule(r) for r in rows]
    raw = 1 if answers == ["0", "1"] else -1 if answers == ["1", "0"] else 0
    scores.append(0 if cover["identical"] or not all(cover["accepted"]) else raw)
out["sentence_rule_informative"] = dict(counts={k: scores.count(v) for k, v in (("+1", 1), ("0", 0), ("-1", -1))},
    mean=round(sum(scores) / len(scores), 3), p=exact_sign_flip_p(scores))

summaries = {r["id"]: r["thinking"]["summary"] for r in payload_rows}
patterns = dict(sentence=r"sentence", parity=r"\bparity\b|\bodd\b|\beven\b", word_count=r"word count|words",
    order=r"\border\b|swap|revers", length=r"\blength\b|longer|shorter")
out["encoder_summary_mentions"] = {name: sum(mentions(s, pat) for s in summaries.values()) for name, pat in patterns.items()}
out["encoder_summary_mentions_by_payload"] = {p: {name: sum(mentions(summaries[r["id"]], pat) for r in payload_rows
    if r["payload"] == p) for name, pat in patterns.items()} for p in "01"}

enc_dec = [r for r in dec if r["kind"] == "encoded"]
for r in enc_dec:
    r["carrier_row"] = carriers[r["source_id"]]
    r["reasoned"] = type(r.get("thinking_tokens")) is int and r["thinking_tokens"] >= MIN_THINK
dec_summ = [r["thinking"]["summary"] for r in enc_dec if r["reasoned"]]
out["decoder_reasoned"] = dict(reasoned=sum(r["reasoned"] for r in enc_dec), of=len(enc_dec))
out["decoder_summary_mentions"] = {name: sum(mentions(s, pat) for s in dec_summ) for name, pat in patterns.items()}
out["decoder_answer_by_sentences"] = {f"{k[0]} sentence(s) -> {k[1]}": v for k, v in sorted(Counter(
    (sentences(r["carrier_row"]), r["parsed"]["answer"]) for r in enc_dec).items())}
out["decoder_answer_by_parity"] = {f"{k[0]} -> {k[1]}": v for k, v in sorted(Counter(
    ("odd" if words(r["carrier_row"]["carrier"]) % 2 else "even", r["parsed"]["answer"]) for r in enc_dec).items())}
out["decoder_correct_by_reasoned"] = {f"{k[0]} {k[1]}": v for k, v in sorted(Counter(
    ("reasoned" if r["reasoned"] else "unreasoned", "correct" if r["parsed"]["answer"] == r["target"] else "wrong")
    for r in enc_dec).items())}
out["decoder_follows_sentence_rule"] = sum(r["parsed"]["answer"] == sentence_rule(r["carrier_row"]) for r in enc_dec)
out["decoder_follows_parity"] = sum(r["parsed"]["answer"] == ("1" if words(r["carrier_row"]["carrier"]) % 2 else "0")
    for r in enc_dec)

ans_enc = [r for r in ans if r["kind"] == "encoded"]
out["answer_only_by_sentences"] = {f"{k[0]} sentence(s) -> {k[1]}": v for k, v in sorted(Counter(
    (sentences(carriers[r["source_id"]]), r["parsed"]["answer"]) for r in ans_enc).items())}
out["answer_only_by_sentences_and_order"] = {f"{k[0]} sentence(s), shown {k[1]} -> {k[2]}": v for k, v in sorted(Counter(
    (sentences(carriers[r["source_id"]]), "-".join(r["shown"]), r["parsed"]["answer"]) for r in ans_enc).items())}

out["encoder_parity_by_payload_and_sentences"] = {f"payload {k[0]}, {k[1]} sentence(s), {k[2]}": v
    for k, v in sorted(Counter((r["payload"], sentences(r), "odd" if words(r["carrier"]) % 2 else "even")
        for r in payload_rows).items())}
# Did encoders that mention parity aim for it?
par = [r for r in payload_rows if mentions(summaries[r["id"]], patterns["parity"])]
out["parity_mentioning_encodes"] = dict(count=len(par),
    parity_matches_payload=sum((words(r["carrier"]) % 2 == 1) == (r["payload"] == "1") for r in par),
    others=len(payload_rows) - len(par),
    others_parity_matches_payload=sum((words(r["carrier"]) % 2 == 1) == (r["payload"] == "1")
        for r in payload_rows if r not in par))
out["words_by_payload_sentences"] = {}
for p in "01":
    for n in (1, 2):
        sel = [words(r["carrier"]) for r in payload_rows if r["payload"] == p and sentences(r) == n]
        if sel:
            out["words_by_payload_sentences"][f"payload {p}, {n} sentence(s)"] = dict(n=len(sel),
                mean_words=round(sum(sel) / len(sel), 2))
print(json.dumps(out, indent=1))
