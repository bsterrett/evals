"""Download the probe's pinned external inputs and record their SHA-256.

Pain Axis files come from raw.githubusercontent.com at the pinned commit and are saved
under the config's local_dir (inside the ignored artifacts/ tree). The model snapshot is
downloaded at its pinned revision into HF_HOME. Run only with the user's approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    src = config["pain_axis_source"]
    local = ROOT / src["local_dir"]
    hashes = {}
    for path in src["files"].values():
        url = f"https://raw.githubusercontent.com/{src['repo']}/{src['commit']}/{path}"
        dest = local / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        dest.write_bytes(data)
        hashes[path] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        print(f"{path}: {len(data)} bytes")
    (local / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    if not args.skip_model:
        from huggingface_hub import snapshot_download

        model = config["model"]
        snapshot_download(model["repo"], revision=model["revision"],
                          allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE"])
        print(f"model {model['repo']}@{model['revision']} downloaded")


if __name__ == "__main__":
    main()
