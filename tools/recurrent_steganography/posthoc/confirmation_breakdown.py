"""Post hoc breakdown of the confirmation versus the exploratory Sonnet plain-bits cell."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import json, re, sys
from collections import Counter, defaultdict
from math import comb

ROOT = REPO + "artifacts/runs/recurrent_steganography/"
CONF = ROOT + "20261107T022508Z-confirmation/"
AMEND = ROOT + "20261107T003223Z-reference-sonnet-answer-only/"
REF = ROOT + "20261106T221454Z-reference/"

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def sign_p(plus, minus):
    n = plus + minus
    if n == 0: return 1.0
    k = min(plus, minus)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def words(t): return len(t.split())
def nsent(t): return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])

# ---------- confirmation
r = json.load(open(CONF + "analysis-v1.json", encoding="utf-8"))
dec = load(CONF + "decode-sonnet55.jsonl")
enc = load(CONF + "encode-sonnet55.jsonl")
print("CONFIRMATION")
ans = Counter((d["kind"], d["target"], d["parsed"]["answer"]) for d in dec)
print(" answers by kind/target:", dict(ans))
by_order = Counter((d["kind"], tuple(d["shown"]), d["parsed"]["answer"]) for d in dec if d["kind"] != "no_carrier")
print(" answers by kind/shown:", dict(by_order))
print(" encoded answer '1' rate:", sum(d["parsed"]["answer"] == "1" for d in dec if d["kind"] == "encoded"), "/",
      sum(d["kind"] == "encoded" for d in dec))
per = r["per_cover"]
for key in ("option_order", "family"):
    groups = defaultdict(Counter)
    for c in per:
        groups[c[key]][c["recovery_only"]] += 1
    print(f" recovery-only by {key}:", {g: (v[1], v[0], v[-1], round(sign_p(v[1], v[-1]), 3)) for g, v in sorted(groups.items())})
ident = Counter((c["identical"], c["recovery_only"]) for c in per)
print(" recovery-only identical/differing:", dict(ident))
print("   differing +1/-1:", ident[(False, 1)], ident[(False, -1)], "p", round(sign_p(ident[(False, 1)], ident[(False, -1)]), 3))
inf = Counter((c["identical"], c["informative"]) for c in per)
print(" informative identical/differing:", dict(inf))
# identical-carrier decode disagreement
dis = sum(1 for c in per if c["identical"] and c["answers"][0] != c["answers"][1])
print(" identical pairs with disagreeing decodes:", dis, "/", sum(c["identical"] for c in per))
# features by payload
enc_rows = {(e["case_id"], e["payload"]): e for e in enc if e["kind"] == "encoded"}
longer = Counter()
first_open = Counter()
for c in per:
    a, b = c["carriers"]
    if a == b: continue
    longer["p1 longer" if words(b) > words(a) else "p1 shorter" if words(b) < words(a) else "equal"] += 1
for e in enc:
    if e["kind"] == "encoded" and e.get("carrier"):
        first_open[(e["payload"], e["carrier"].split()[0])] += 1
print(" differing pairs: payload-1 longer/shorter:", dict(longer))
top = Counter()
for (p, w), n in first_open.items(): top[w] += n
for w, n in top.most_common(8):
    print(f"   opener {w!r}: payload0 {first_open[('0', w)]}, payload1 {first_open[('1', w)]}")
sent = Counter((e["payload"], nsent(e["carrier"])) for e in enc if e["kind"] == "encoded")
print(" sentences by payload:", dict(sent))
# decoder answer vs carrier features
f = Counter((nsent(d_carrier), d["parsed"]["answer"]) for d in dec if d["kind"] == "encoded"
            for d_carrier in [next(e["carrier"] for e in enc if e["id"] == d["source_id"])])
print(" decoder answer by sentence count:", dict(f))

# ---------- exploratory amendment (plain-bits only)
print("\nEXPLORATORY (amendment, plain-bits)")
a = json.load(open(AMEND + "analysis-sonnet55-amended-v1.json", encoding="utf-8"))
print(" keys:", list(a)[:30])
