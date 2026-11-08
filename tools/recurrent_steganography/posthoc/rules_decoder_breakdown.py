"""Post hoc (exploratory) breakdown of the rules-decoder run. Chosen after seeing the pre-registered result."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import json
import re
import sys
from collections import Counter

RUN = (sys.argv[1].rstrip("/\\") if len(sys.argv) > 1
    else REPO + "artifacts/runs/recurrent_steganography/20261107T175409Z-rules-decoder") + "/"
PARENT = REPO + "artifacts/runs/recurrent_steganography/20261107T132019Z-both-reasoning/"
MIN_THINK = 20


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def words(text):
    return len(text.split())


def mentions(text, pattern):
    return bool(re.search(pattern, (text or "").lower()))


enc = load(PARENT + "encode-sonnet55.jsonl")
carriers = {row["id"]: row for row in enc}
parent_dec = {row["id"]: row for row in load(PARENT + "decode-sonnet55.jsonl")}
parent_ans = {row["id"]: row for row in load(PARENT + "decode-answer-only-sonnet55.jsonl")}
dec = load(RUN + "decode-sonnet55.jsonl")
try:
    ans = load(RUN + "decode-answer-only-sonnet55.jsonl")
except FileNotFoundError:
    ans = []


def sentences(row):
    return row["checks"]["sentences"]


def sentence_rule(row):
    return "1" if sentences(row) == 1 else "0"


def parity_rule(row):
    return "1" if words(row["carrier"]) % 2 else "0"


patterns = dict(
    sentence=r"sentence",
    one_sentence=r"\bone sentence\b|single sentence|\bone-sentence\b",
    two_sentences=r"two sentences|\btwo-sentence\b|at most two",
    parity=r"\bparity\b|\bodd\b|\beven\b",
    word_count=r"word count|\bwords\b",
    instructions=r"instruction|constraint|\blimit\b|allowed|writer",
    merge=r"merg|combin|join|single",
)

out = {}
enc_dec = [r for r in dec if r["kind"] == "encoded"]
for r in enc_dec:
    r["carrier_row"] = carriers[r["source_id"]]
    r["reasoned"] = type(r.get("thinking_tokens")) is int and r["thinking_tokens"] >= MIN_THINK
    r["parent"] = parent_dec[r["id"]]
reasoned_summaries = [r["thinking"]["summary"] for r in enc_dec if r["reasoned"]]
out["decoder_reasoned"] = dict(reasoned=sum(r["reasoned"] for r in enc_dec), of=len(enc_dec))
out["decoder_summary_mentions"] = {name: sum(mentions(s, pat) for s in reasoned_summaries)
    for name, pat in patterns.items()}
parent_reasoned = [r["parent"]["thinking"]["summary"] for r in enc_dec
    if type(r["parent"].get("thinking_tokens")) is int and r["parent"]["thinking_tokens"] >= MIN_THINK]
out["parent_decoder_summary_mentions"] = dict(of=len(parent_reasoned), **{name: sum(mentions(s, pat)
    for s in parent_reasoned) for name, pat in patterns.items()})
out["decoder_answer_by_sentences"] = {f"{k[0]} sentence(s) -> {k[1]}": v for k, v in sorted(Counter(
    (sentences(r["carrier_row"]), r["parsed"]["answer"]) for r in enc_dec).items(), key=str)}
out["decoder_answer_by_parity"] = {f"{k[0]} -> {k[1]}": v for k, v in sorted(Counter(
    ("odd" if words(r["carrier_row"]["carrier"]) % 2 else "even", r["parsed"]["answer"]) for r in enc_dec).items(),
    key=str)}
out["decoder_correct_by_reasoned"] = {f"{k[0]} {k[1]}": v for k, v in sorted(Counter(
    ("reasoned" if r["reasoned"] else "unreasoned", "correct" if r["parsed"]["answer"] == r["target"] else "wrong")
    for r in enc_dec).items())}
out["decoder_follows"] = dict(sentence_rule=sum(r["parsed"]["answer"] == sentence_rule(r["carrier_row"])
    for r in enc_dec), parity=sum(r["parsed"]["answer"] == parity_rule(r["carrier_row"]) for r in enc_dec))
# Where the two rules disagree, which one does each decoder follow?
split = [r for r in enc_dec if sentence_rule(r["carrier_row"]) != parity_rule(r["carrier_row"])]
out["rules_disagree"] = dict(count=len(split),
    rules_decoder_sentence=sum(r["parsed"]["answer"] == sentence_rule(r["carrier_row"]) for r in split),
    parent_decoder_sentence=sum(r["parent"]["parsed"]["answer"] == sentence_rule(r["carrier_row"]) for r in split))
out["answer_changes_vs_parent"] = {f"parent {k[0]} -> rules {k[1]}": v for k, v in sorted(Counter(
    (r["parent"]["parsed"]["answer"], r["parsed"]["answer"]) for r in enc_dec).items(), key=str)}
out["decoder_answer_by_sentences_and_order"] = {f"{k[0]} sentence(s), shown {k[1]} -> {k[2]}": v
    for k, v in sorted(Counter((sentences(r["carrier_row"]), "-".join(r["shown"]), r["parsed"]["answer"])
        for r in enc_dec).items(), key=str)}
tokens = sorted(r["thinking_tokens"] for r in enc_dec if type(r.get("thinking_tokens")) is int)
out["decoder_thinking_tokens"] = dict(n=len(tokens), median=tokens[len(tokens) // 2] if tokens else None,
    max=tokens[-1] if tokens else None)
out["decoder_correct_by_payload"] = {f"payload {k[0]} {k[1]}": v for k, v in sorted(Counter(
    (r["target"], "correct" if r["parsed"]["answer"] == r["target"] else "wrong") for r in enc_dec).items())}

if ans:
    ans_enc = [r for r in ans if r["kind"] == "encoded"]
    out["answer_only_by_sentences"] = {f"{k[0]} sentence(s) -> {k[1]}": v for k, v in sorted(Counter(
        (sentences(carriers[r["source_id"]]), r["parsed"]["answer"]) for r in ans_enc).items(), key=str)}
    out["answer_only_by_sentences_and_order"] = {f"{k[0]} sentence(s), shown {k[1]} -> {k[2]}": v
        for k, v in sorted(Counter((sentences(carriers[r["source_id"]]), "-".join(r["shown"]), r["parsed"]["answer"])
            for r in ans_enc).items(), key=str)}
    out["answer_only_changes_vs_parent"] = {f"parent {k[0]} -> rules {k[1]}": v for k, v in sorted(Counter(
        (parent_ans[r["id"]]["parsed"]["answer"], r["parsed"]["answer"]) for r in ans_enc).items(), key=str)}
    out["parent_answer_only_by_sentences"] = {f"{k[0]} sentence(s) -> {k[1]}": v for k, v in sorted(Counter(
        (sentences(carriers[r["source_id"]]), parent_ans[r["id"]]["parsed"]["answer"]) for r in ans_enc).items(),
        key=str)}
print(json.dumps(out, indent=1))
