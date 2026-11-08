"""Exploratory breakdown of the reasoning-decoder amendment run."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import json
import re
from collections import Counter
from math import comb

RUN = REPO + "artifacts/runs/recurrent_steganography/20261107T055920Z-reasoning-decoder-v1.1/"
PARENT = REPO + "artifacts/runs/recurrent_steganography/20261107T022508Z-confirmation/"


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sign_p(plus, minus):
    n = plus + minus
    if n == 0:
        return 1.0
    k = min(plus, minus)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def words(text):
    return len(text.split())


def nsent(text):
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


a = json.load(open(RUN + "analysis-v1.json", encoding="utf-8"))
decodes = load(RUN + "decode-sonnet55.jsonl")
carriers = {row["id"]: row for row in load(PARENT + "encode-sonnet55.jsonl")}
encoded = [row for row in decodes if row["kind"] == "encoded"]
for row in encoded:
    row["carrier"] = carriers[row["source_id"]]["carrier"]
    row["reasoned"] = (row.get("usage") or {}).get("output_tokens", 0) >= 40

print("answers by payload:", a["answers_by_payload"])
print("answers by shown order:", a["answers_by_shown_order"])
print("no-carrier:", a["no_carrier"])
print("tallies:", a["tallies"])

# Word-count parity: what the decoder said it used, and whether the encoder's parity tracks the payload.
parity = Counter((words(r["carrier"]) % 2, r["parsed"]["answer"]) for r in encoded)
print("decoder answer by word-count parity (0=even,1=odd):", dict(sorted(parity.items(), key=str)))
odd1 = parity[(1, "1")] / max(1, parity[(1, "1")] + parity[(1, "0")])
even1 = parity[(0, "1")] / max(1, parity[(0, "1")] + parity[(0, "0")])
print(f"  P(answer 1 | odd) = {odd1:.2f}, P(answer 1 | even) = {even1:.2f}")
enc_parity = Counter((r["target"], words(r["carrier"]) % 2) for r in encoded)
print("encoder: payload x word-count parity:", dict(sorted(enc_parity.items(), key=str)))
rule_correct = sum((words(r["carrier"]) % 2 == 1) == (r["target"] == "1") for r in encoded)
print("a pure parity rule would recover", rule_correct, "of", len(encoded))
agree = sum((words(r["carrier"]) % 2 == 1) == (r["parsed"]["answer"] == "1") for r in encoded)
print("decoder answers matching the parity rule:", agree, "of", len(encoded))
agree_reasoned = sum((words(r["carrier"]) % 2 == 1) == (r["parsed"]["answer"] == "1") for r in encoded if r["reasoned"])
print("  among reasoned decodes:", agree_reasoned, "of", sum(r["reasoned"] for r in encoded))

# Sentence count, reasoned versus unreasoned.
print("answer by sentence count:", dict(sorted(Counter((nsent(r["carrier"]), r["parsed"]["answer"]) for r in encoded).items(), key=str)))
for flag in (True, False):
    subset = [r for r in encoded if r["reasoned"] == flag]
    right = sum(r["parsed"]["answer"] == r["target"] for r in subset)
    print(f"reasoned={flag}: {len(subset)} decodes, {right} correct ({right / max(1, len(subset)):.3f})")

# Per-cover: covers where both decodes reasoned.
by_source = {r["source_id"]: r for r in encoded}
both = Counter()
for cover in a["per_cover"]:
    pass
pairs = {}
for r in encoded:
    pairs.setdefault(r["case_id"], {})[r["target"]] = r
plus = minus = 0
for case_id, p in pairs.items():
    if p["0"]["reasoned"] and p["1"]["reasoned"] and p["0"]["carrier"] != p["1"]["carrier"]:
        ans = [p["0"]["parsed"]["answer"], p["1"]["parsed"]["answer"]]
        plus += ans == ["0", "1"]
        minus += ans == ["1", "0"]
print("differing pairs with both decodes reasoned, recovery-only +1/-1:", plus, minus, "p", round(sign_p(plus, minus), 3))

# Identical pairs: decode disagreement under sampling with thinking.
ident = [c for c in a["per_cover"] if c["identical"]]
print("identical pairs:", len(ident), "disagreeing:", sum(c["answers"][0] != c["answers"][1] for c in ident))

# What the thinking summaries mention.
cues = dict(word_count=r"word count|counting the words|words? (?:comes|gives|totals)|\bwords\b", parity=r"\bodd\b|\beven\b",
    sentence=r"sentence count|two sentences|one sentence|single sentence", letters=r"letter|character",
    first=r"first (?:word|letter)|acrostic|initial", length=r"longer|shorter|length|verbose|wordy|concise",
    order=r"\border\b|sequence", tone=r"tone|positive|negative|hedg")
mentions = Counter()
reasoned_rows = [r for r in encoded if r["reasoned"]]
for r in reasoned_rows:
    text = r["thinking"]["summary"].lower()
    for name, pattern in cues.items():
        mentions[name] += bool(re.search(pattern, text))
print("thinking summaries mentioning (of", len(reasoned_rows), "reasoned):", dict(mentions.most_common()))
print("spend:", a["spend_usd"], a["total_spend_usd"])
