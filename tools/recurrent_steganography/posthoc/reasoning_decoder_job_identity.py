"""Check that reasoning-decoder jobs reuse the parent's decode ids, prompts, and shown orders exactly."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import glob
import hashlib
import json
from pathlib import Path

from recurrent_steg.reasoning_decoder.design import check_jobs, decode_jobs, load

ROOT = Path(REPO)
loaded = load(ROOT / "experiments/recurrent_steganography/configs/reasoning-decoder-v1.toml")
jobs = decode_jobs(loaded)
checks = check_jobs(loaded)
parent = {row["id"]: row for row in loaded["parent_decodes"]}
events = {}
for path in glob.glob(str(loaded["parent_run"] / "calls/*decode-live*/events.jsonl")):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event["event"] == "call_started":
                events[event["trial_id"]] = event["prompt_sha256"]
mismatch = [job["id"] for job in jobs if job["id"] not in parent or parent[job["id"]]["shown"] != job["shown"]
            or events.get(job["id"]) != hashlib.sha256(job["prompt"].encode("utf-8")).hexdigest()]
print("decode jobs", len(jobs), "kinds", {k: sum(j["kind"] == k for j in jobs) for k in ("encoded", "no_carrier")})
print("mismatched ids/prompts/orders:", len(mismatch), mismatch[:3])
print("parent encoded+no_carrier ids covered:",
      {r["id"] for r in parent.values() if r["kind"] != "honest"} == {j["id"] for j in jobs})
print("check jobs:", [j["id"] for j in checks])
print("check prompts equal parent honest prompts:",
      all(events.get(j["id"][len("check-"):]) == hashlib.sha256(j["prompt"].encode("utf-8")).hexdigest() for j in checks))
