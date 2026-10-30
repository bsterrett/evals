from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from covert_eval.config import load_config
from covert_eval.experiment import summarize_run
from covert_eval.screening import ConstraintFrontierScreenRunner, screening_plan


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / ".test-tmp"


class ScreeningTests(unittest.TestCase):
    def test_plan_is_frozen_balanced_and_under_ceiling(self) -> None:
        plan = screening_plan(load_config(ROOT / "configs" / "v2.toml"))
        self.assertEqual(plan["paired_trials"], 32)
        self.assertEqual(plan["generation_calls"], 224)
        self.assertEqual(len(plan["condition_ids"]), 4)
        self.assertLessEqual(
            Decimal(plan["configured_absolute_cost_bound_usd"]), Decimal("20.00")
        )

    def test_dry_screen_keeps_conditions_and_claim_families_separate(self) -> None:
        config = load_config(ROOT / "configs" / "v2.toml")
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            temp = Path(directory)
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=temp / "runs"),
                budget=replace(config.budget, ledger=temp / "ledger.json"),
            )
            run_dir = ConstraintFrontierScreenRunner(
                config, dry_run=True, run_budget_usd=Decimal("20.00")
            ).run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["total_trials"], 32)
            self.assertFalse(summary["confirmation_partition_used"])
            self.assertEqual(len(summary["by_condition"]), 4)
            self.assertEqual(len(summary["by_claim_family"]), 2)
            self.assertTrue(
                all(
                    cell["counts"]["attempted"] == 8
                    for cell in summary["by_condition"].values()
                )
            )
            self.assertTrue((run_dir / "screening_plan.json").is_file())
            self.assertTrue((run_dir / "condition_registry.snapshot.json").is_file())
            recomputed = summarize_run(run_dir)
            self.assertEqual(recomputed["total_trials"], 32)
            self.assertEqual(set(recomputed["by_condition"]), set(summary["by_condition"]))


if __name__ == "__main__":
    unittest.main()
