"""Install the isolated WSL runtimes. Run with Ubuntu-LLM's python3, not Windows.

The Python installer uses no sudo and does not change drivers or Ollama. Dory
additionally needs Ubuntu build-essential and python3.12-dev installed first.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
UV_VERSION = "0.9.7"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    parser.add_argument("--profile", default="all")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this installer inside Ubuntu-LLM.")
    root = args.root.resolve()
    for name in ["tools", "envs", "cache/huggingface", "logs", "benchmarks"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    uv = root / "tools/uv"
    if not uv.exists():
        artifact = "uv-x86_64-unknown-linux-gnu.tar.gz"
        url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{artifact}"
        data = urllib.request.urlopen(url, timeout=120).read()
        expected = urllib.request.urlopen(url + ".sha256", timeout=30).read().decode().split()[0]
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("uv archive checksum mismatch")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith("/uv"))
            uv.write_bytes(archive.extractfile(member).read())
        uv.chmod(0o755)
        (root / "tools/uv.sha256").write_text(hashlib.sha256(uv.read_bytes()).hexdigest() + "\n")
    env = dict(os.environ, UV_CACHE_DIR=str(root / "cache/uv"), HF_HOME=str(root / "cache/huggingface"))
    profiles = json.loads((ROOT / "src/eval_core/local/profiles.json").read_text())
    if args.profile not in (*profiles, "all"):
        parser.error("Unknown model profile")
    for name, profile in profiles.items():
        if args.profile not in ("all", name):
            continue
        runtime_name = profile.get("environment", name)
        runtime = root / "envs" / runtime_name
        python = runtime / "bin/python"
        def run(*cmd: str) -> None:
            print("Running:", " ".join(map(str, cmd)), flush=True)
            subprocess.run(list(map(str, cmd)), env=env, check=True)
        if not python.exists():
            run(uv, "venv", "--python", "/usr/bin/python3", runtime)
        lock = ROOT / "local_models/locks" / f"{runtime_name}.txt"
        if lock.exists():
            backend_args = [] if runtime_name == "dory" else ["--torch-backend", "cu128"]
            run(uv, "pip", "sync", "--python", python, "--require-hashes", *backend_args, lock)
        elif runtime_name == "dory":
            run(uv, "pip", "install", "--python", python, f"vllm=={profile['vllm']}")
        else:
            run(uv, "pip", "install", "--python", python, "torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu128")
            run(uv, "pip", "install", "--python", python, f"transformers=={profile['transformers']}",
                "accelerate==1.11.0", "huggingface-hub==0.36.0", "safetensors==0.6.2", "numpy==2.2.6")
        if runtime_name == "dory":
            if not shutil.which("cc") or not Path("/usr/include/python3.12/Python.h").exists():
                raise RuntimeError("Dory requires Ubuntu build-essential and python3.12-dev; see the local setup document")
            run(uv, "pip", "install", "--python", python, "-e", ROOT)
        freeze = subprocess.check_output([str(uv), "pip", "freeze", "--python", str(python)], env=env, text=True)
        (root / "envs" / f"{runtime_name}.freeze.txt").write_text(freeze)
        run(python, "-c", "import torch; assert torch.cuda.is_available(); "
            "x=torch.ones((64,64),device='cuda',dtype=torch.bfloat16); "
            "assert (x@x).isfinite().all(); "
            "print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),torch.cuda.get_device_capability())")
        run(python, ROOT / "tools/local_models/download.py", "--root", root, "--profile", name)


if __name__ == "__main__":
    main()
