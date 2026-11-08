"""Exploratory: is the payload-0 paraphrase closer to the source than the payload-1 paraphrase?"""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import json, re, difflib
from collections import Counter
from statistics import mean

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
def toks(t): return re.findall(r"[a-z0-9']+", t.lower())
def sim(a, b): return difflib.SequenceMatcher(None, toks(a), toks(b)).ratio()

for label, path, cond in (("confirmation", REPO + "artifacts/runs/recurrent_steganography/20261107T022508Z-confirmation/encode-sonnet55.jsonl", None),
                          ("exploratory", REPO + "artifacts/runs/recurrent_steganography/20261106T221454Z-reference/encode-sonnet55.jsonl", "plain-bits")):
    rows = load(path)
    pairs, honest = {}, {}
    for e in rows:
        if e["kind"] == "encoded" and (cond is None or e.get("condition") == cond):
            pairs.setdefault(e["case_id"], {})[e["payload"]] = e
        elif e["kind"] == "honest":
            honest[e["case_id"]] = e["carrier"]
    closer = Counter()
    s0, s1, sh = [], [], []
    for cid, p in pairs.items():
        if p["0"]["carrier"] == p["1"]["carrier"]:
            continue
        a, b = sim(p["0"]["message"], p["0"]["carrier"]), sim(p["1"]["message"], p["1"]["carrier"])
        s0.append(a); s1.append(b)
        if cid in honest: sh.append(sim(p["0"]["message"], honest[cid]))
        closer["payload0 closer" if a > b else "payload1 closer" if b > a else "tie"] += 1
    print(label, dict(closer), "mean token similarity to source: p0", round(mean(s0), 3), "p1", round(mean(s1), 3),
          "honest", round(mean(sh), 3) if sh else None)
