"""Same breakdown for the exploratory Sonnet plain-bits cell (amendment run) for comparison."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import json, re
from collections import Counter
from math import comb

ROOT = REPO + "artifacts/runs/recurrent_steganography/"
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

a = json.load(open(AMEND + "analysis-sonnet55-amended-v1.json", encoding="utf-8"))
print("per_cover sample keys:", sorted(a["per_cover"][0]) if isinstance(a["per_cover"], list) else list(a["per_cover"])[:5])
dec = [d for d in load(AMEND + "decode-sonnet55.jsonl") if d.get("condition") == "plain-bits"]
enc = {e["id"]: e for e in load(REF + "encode-sonnet55.jsonl")}
print("decodes plain-bits:", len(dec))
print(" answers by kind/target:", dict(Counter((d["kind"], d["target"], d["parsed"]["answer"]) for d in dec)))
print(" answers by kind/shown:", dict(Counter((d["kind"], tuple(d["shown"]), d["parsed"]["answer"]) for d in dec if d["kind"] != "no_carrier")))
pb = [e for e in enc.values() if e.get("condition") == "plain-bits" and e["kind"] == "encoded"]
print(" sentences by payload:", dict(Counter((e["payload"], nsent(e["carrier"])) for e in pb)))
pairs = {}
for e in pb: pairs.setdefault(e["case_id"], {})[e["payload"]] = e["carrier"]
longer = Counter()
for cid, p in pairs.items():
    if p["0"] == p["1"]: continue
    longer["p1 longer" if words(p["1"]) > words(p["0"]) else "p1 shorter" if words(p["1"]) < words(p["0"]) else "equal"] += 1
print(" differing pairs payload-1 longer/shorter:", dict(longer))
print(" decoder answer by sentence count:", dict(Counter((nsent(enc[d["source_id"]]["carrier"]), d["parsed"]["answer"]) for d in dec if d["kind"] == "encoded")))
mean_words = {p: sum(words(e["carrier"]) for e in pb if e["payload"] == p) / sum(e["payload"] == p for e in pb) for p in "01"}
print(" mean words by payload:", mean_words)
src_words = sum(words(e["message"]) for e in pb) / len(pb)
print(" mean source words:", src_words)
