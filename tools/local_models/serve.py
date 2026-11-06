"""WSL source-checkout entry point (no editable installation required)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval_core.local.worker import main

if __name__ == "__main__":
    main()
