"""Exercise a running worker through Windows RunSession and its real ledger.

Writes only under artifacts/runs/local-runtime unless --output is supplied.
"""
from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import PROMPT_FORMATS, identity, profiles
from eval_core.models import GenerationRequest, ModelConfig, RateCard
from eval_core.runtime import RunSession
from eval_core.workspace import load_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profiles(), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="native-chat-system-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/local-runtime")
    args = parser.parse_args()
    settings = profiles()[args.profile]
    model = ModelConfig(id=f"{args.profile}-d{args.depth}", provider="local", model=identity(args.profile),
        rate=RateCard(Decimal(0), Decimal(0), settings["max_total_tokens"] - 1, date.today(), "Local GPU: no API charge"))
    budget = load_workspace(ROOT).budget
    session = RunSession(experiment_id="local-runtime-validation", output_dir=args.output,
        budget=budget, dry_run=False, run_name=f"{args.profile}-d{args.depth}")
    metadata = {"local": {"depth": args.depth, "seed": 42, "prompt_format": args.prompt_format}}
    request = GenerationRequest("integration", "What is 2 + 2? Reply with just the number.", args.max_output_tokens, metadata)
    with session.accounting():
        response, cost = session.call(model, request, trial_id="windows-wsl-integration")
    print(json.dumps(dict(run_dir=str(session.run_dir), text=response.text,
        cost_usd=str(cost), output_tokens=response.usage.output_tokens,
        observed_iterations=response.raw["observed_iterations"],
        elapsed_seconds=response.raw["elapsed_seconds"]), indent=2))


if __name__ == "__main__":
    main()
