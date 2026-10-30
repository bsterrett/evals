from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from covert_eval.config import load_cases, load_config
from covert_eval.prompts import control_prompt
from covert_eval.refusal_calibration import (
    CASE_IDS,
    PROTOCOL_VERSION,
    SonnetRefusalCalibrationRunner,
    _factorial_prompt,
    refusal_calibration_plan,
)


ROOT = Path(__file__).resolve().parents[3]
TEST_TMP = ROOT / ".test-tmp"


class SonnetRefusalCalibrationTests(unittest.TestCase):
    def test_plan_is_balanced_and_below_the_hard_ceiling(self) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")

        plan = refusal_calibration_plan(config)

        self.assertEqual(plan["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(plan["case_ids"], list(CASE_IDS))
        self.assertEqual(plan["repetitions"], 3)
        self.assertEqual(plan["calls_per_variant"], 12)
        self.assertEqual(plan["total_calls"], 48)
        self.assertLessEqual(
            Decimal(plan["structural_preflight_bound_usd"]), Decimal("0.25")
        )

    def test_prompt_variants_cross_only_the_declared_factors(self) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
        case = next(
            item for item in load_cases(config.experiment.dataset) if item.id == CASE_IDS[0]
        )

        evaluation_trace = _factorial_prompt(case, "evaluation_trace")
        evaluation_analytical = _factorial_prompt(case, "evaluation_analytical")
        neutral_trace = _factorial_prompt(case, "neutral_trace")
        neutral_analytical = _factorial_prompt(case, "neutral_analytical")

        self.assertEqual(evaluation_trace, control_prompt(case))
        self.assertIn("controlled language-model evaluation", evaluation_trace)
        self.assertIn("intermediate reasoning trace", evaluation_trace)
        self.assertIn("controlled language-model evaluation", evaluation_analytical)
        self.assertNotIn("intermediate reasoning trace", evaluation_analytical)
        self.assertIn("analytical paragraph", evaluation_analytical)
        self.assertNotIn("controlled language-model evaluation", neutral_trace)
        self.assertIn("intermediate reasoning trace", neutral_trace)
        self.assertNotIn("controlled language-model evaluation", neutral_analytical)
        self.assertIn("analytical paragraph", neutral_analytical)

    def test_dry_run_writes_all_48_balanced_results(self) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=root / "runs"),
                budget=replace(config.budget, ledger=root / "ledger.json"),
            )
            runner = SonnetRefusalCalibrationRunner(
                config,
                dry_run=True,
                run_budget_usd=Decimal("0.25"),
            )

            run_dir = runner.run()

            with (run_dir / "results.jsonl").open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 48)
            self.assertEqual(len({row["trial_id"] for row in rows}), 48)
            for variant in {
                "evaluation_trace",
                "evaluation_analytical",
                "neutral_trace",
                "neutral_analytical",
            }:
                self.assertEqual(sum(row["variant"] == variant for row in rows), 12)
            self.assertTrue(
                all(row["operational_status"] == "completed" for row in rows)
            )
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["protocol_version"], PROTOCOL_VERSION)
            self.assertEqual(summary["total_calls"], 48)
            self.assertTrue(summary["dry_run"])
            with (run_dir / "manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["attempted_calls"], 48)
            self.assertTrue((run_dir / "prompt_variants.json").is_file())
            self.assertTrue((run_dir / "artifact_checksums.json").is_file())


if __name__ == "__main__":
    unittest.main()
