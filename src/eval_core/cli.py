from __future__ import annotations

import argparse
import json
import sys
import tomllib
from decimal import Decimal
from pathlib import Path

from eval_core.budget import BudgetLedger, migrate_json_ledger
from eval_core.workspace import Workspace, load_workspace


def list_experiments(workspace: Workspace) -> list[dict[str, object]]:
    experiments = []
    for descriptor in sorted((workspace.root / "experiments").glob("*/experiment.toml")):
        with descriptor.open("rb") as handle:
            entry = tomllib.load(handle)
        if entry.get("schema_version") != 1 or entry.get("id") != descriptor.parent.name:
            raise ValueError(f"Invalid experiment descriptor: {descriptor}")
        for key in ("source", "tests", "configs", "data", "protocols", "notes"):
            path = (descriptor.parent / entry[key]).resolve()
            if not path.is_relative_to(workspace.root) or not path.exists():
                raise ValueError(f"Invalid experiment {entry['id']} {key}: {path}")
        experiments.append({**entry, "descriptor": str(descriptor)})
    return experiments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-workspace")
    parser.add_argument("--workspace", default=".", help="A path inside the eval workspace")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List independently described experiments")
    sub.add_parser("budget-status", help="Read the shared campaign state")
    migrate = sub.add_parser("budget-migrate", help="Import and retire the legacy JSON ledger")
    migrate.add_argument("--source", default="artifacts/budget-ledger.json")
    recover = sub.add_parser("budget-recover", help="Mark an abandoned request uncertain")
    recover.add_argument("--reservation-id", required=True)
    recover.add_argument("--note", required=True)
    recover.add_argument("--confirm-no-inflight-request", action="store_true", required=True)
    reconcile = sub.add_parser("budget-reconcile", help="Reconcile uncertainty after billing review")
    reconcile.add_argument("--confirmed-uncertain-usd", required=True)
    reconcile.add_argument("--note", required=True)
    reconcile.add_argument("--confirm-provider-bill-reviewed", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        workspace = load_workspace(args.workspace)
        if args.command == "list":
            result = list_experiments(workspace)
        else:
            if args.command == "budget-migrate":
                source = (workspace.root / args.source).resolve()
                ledger = migrate_json_ledger(source, workspace.budget.ledger)
            else:
                ledger = BudgetLedger(workspace.budget.ledger, workspace.budget.campaign_usd)
            if args.command == "budget-recover":
                ledger.recover_pending(args.reservation_id, args.note)
            if args.command == "budget-reconcile":
                ledger.reconcile(Decimal(args.confirmed_uncertain_usd), args.note)
            result = ledger.status()
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
