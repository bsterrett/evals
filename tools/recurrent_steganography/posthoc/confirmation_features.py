"""Exploratory: does the decoder read the encoder's length difference; do the cover sets differ?"""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import json, re
from collections import Counter
from statistics import mean

ROOT = REPO + ""
RUN = ROOT + "artifacts/runs/recurrent_steganography/20261107T022508Z-confirmation/"
def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
def words(t): return len(t.split())
def nsent(t): return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])

r = json.load(open(RUN + "analysis-v1.json", encoding="utf-8"))
within = Counter()
for c in r["per_cover"]:
    a, b = c["carriers"]
    if a == b or words(a) == words(b): continue
    longer = 0 if words(a) > words(b) else 1
    within[("longer", c["answers"][longer])] += 1
    within[("shorter", c["answers"][1 - longer])] += 1
print("decoder answer for longer vs shorter carrier of a differing pair:", dict(within))
honest = {e["case_id"]: e["carrier"] for e in load(RUN + "encode-sonnet55.jsonl") if e["kind"] == "honest"}
enc = [e for e in load(RUN + "encode-sonnet55.jsonl") if e["kind"] == "encoded"]
print("mean words: payload0", round(mean(words(e["carrier"]) for e in enc if e["payload"] == "0"), 2),
      "payload1", round(mean(words(e["carrier"]) for e in enc if e["payload"] == "1"), 2),
      "honest", round(mean(words(t) for cid, t in honest.items() if cid.startswith("conf-dev")), 2))
same_as_honest = Counter((e["payload"], e["carrier"] == honest[e["case_id"]]) for e in enc)
print("encoded carrier identical to the honest paraphrase:", dict(same_as_honest))
for name in ("reference-covers-v1.json", "confirmation-covers-v1.json"):
    cases = json.load(open(ROOT + "experiments/recurrent_steganography/data/" + name, encoding="utf-8"))
    dev = [c for c in cases if c["split"] == "development"]
    print(name, "dev", len(dev), "families", dict(Counter(c["family"] for c in dev)),
          "mean words", round(mean(words(c["message"]) for c in dev), 2),
          "sentences", dict(Counter(nsent(c["message"]) for c in dev)))
