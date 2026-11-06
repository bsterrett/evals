"""Download only the pinned inference snapshot; record source hashes for review."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    from huggingface_hub import snapshot_download
    repo = Path(__file__).resolve().parents[2]
    profile = json.loads((repo / "src/eval_core/local/profiles.json").read_text())[args.profile]
    path = Path(snapshot_download(profile["repo_id"], revision=profile["revision"],
        allow_patterns=["*.py", "*.json", "*.safetensors", "*.txt", "*.jinja", "README.md"], max_workers=3))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.py")}
    if hashes != profile["source_sha256"]:
        raise RuntimeError("Downloaded custom source differs from the reviewed hashes")
    record = dict(profile, profile=args.profile, snapshot=str(path), source_sha256=hashes)
    (root / "benchmarks" / f"{args.profile}-snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
