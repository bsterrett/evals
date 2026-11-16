"""WSL source-checkout entry point (no editable installation required).

Retrofit parents are served by the retrofit worker; every other profile by the
recurrent-model worker. Both use the same HTTP contract and GPU lock.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval_core.local.protocol import profiles


def selected_family() -> str | None:
    for index, value in enumerate(sys.argv):
        name = value.split("=", 1)[1] if value.startswith("--profile=") else (
            sys.argv[index + 1] if value == "--profile" and index + 1 < len(sys.argv) else None)
        if name in profiles():
            return profiles()[name].get("family", name)
    return None


if __name__ == "__main__":
    if selected_family() in ("qwen3", "loopus"):
        from recurrent_retrofit.worker import main
    else:
        from eval_core.local.worker import main
    main()
