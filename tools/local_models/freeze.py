"""Capture exact tested Linux dependencies with distribution hashes."""
from pathlib import Path
import argparse
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    args = parser.parse_args()
    uv = args.root / "tools/uv"
    locks = ROOT / "local_models/locks"
    locks.mkdir(parents=True, exist_ok=True)
    for profile in ["huginn", "ouro", "dory"]:
        python = args.root / "envs" / profile / "bin/python"
        freeze = subprocess.check_output([str(uv), "pip", "freeze", "--python", str(python)], text=True)
        freeze = "\n".join(line for line in freeze.splitlines() if not line.startswith("-e ")) + "\n"
        source = args.root / "envs" / f"{profile}.freeze.txt"
        source.write_text(freeze)
        command = [str(uv), "pip", "compile", str(source), "--python", str(python),
            *([] if profile == "dory" else ["--torch-backend", "cu128"]),
            "--generate-hashes", "--no-annotate", "--no-header",
            "--output-file", str(locks / f"{profile}.txt")]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        print(f"Locked {profile}: {len(freeze.splitlines())} distributions", flush=True)


if __name__ == "__main__":
    main()
