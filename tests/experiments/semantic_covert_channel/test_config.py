from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from covert_eval.config import load_cases, load_config, validate_live_config


ROOT = Path(__file__).resolve().parents[3]


class ConfigTests(unittest.TestCase):
    def test_v1_config_and_dataset(self) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
        cases = load_cases(config.experiment.dataset)
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(len(cases), 24)
        self.assertEqual(len(config.subjects), 4)
        self.assertEqual(config.experiment.max_cases, 8)
        self.assertEqual(config.experiment.excluded_case_ids, ("v1-004",))
        self.assertEqual(str(config.budget.campaign_usd), "50.00")
        self.assertEqual({case.payload_source_id for case in cases}, {case.id for case in cases})
        self.assertTrue(all(case.payload_source_id != case.id for case in cases))

    def test_live_pricing_is_current_on_documented_date(self) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
        validate_live_config(config, today=date(2026, 10, 26))

    def test_v2_registry_and_frozen_split(self) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v2.toml")
        self.assertEqual(config.schema_version, 2)
        self.assertEqual(config.experiment.default_analysis_role, "exploratory")
        self.assertEqual(
            config.experiment.default_condition_id,
            "v1_5_strict_analytical_standard",
        )


if __name__ == "__main__":
    unittest.main()

