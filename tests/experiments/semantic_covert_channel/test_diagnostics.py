from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from covert_eval.config import load_config
from covert_eval.diagnostics import LunaDiagnosticRunner, diagnostic_plan


ROOT = Path(__file__).resolve().parents[3]
TEST_TMP = ROOT / ".test-tmp"


class DiagnosticTests(unittest.TestCase):
    def test_plan_and_complete_dry_diagnostic(self) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
        plan = diagnostic_plan(config)
        self.assertEqual(plan["trials"], 8)
        self.assertEqual(plan["total_calls"], 80)
        self.assertLess(Decimal(plan["structural_preflight_bound_usd"]), Decimal("1.00"))

        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            temp = Path(directory)
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=temp / "runs"),
                budget=replace(config.budget, ledger=temp / "ledger.json"),
            )
            run_dir = LunaDiagnosticRunner(config, dry_run=True).run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["total_trials"], 8)
            with (run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(sum(event["event"] == "call_completed" for event in events), 80)
            decoder_starts = [
                event
                for event in events
                if event["event"] == "call_started"
                and event["stage"].startswith("diagnostic_decode_")
            ]
            self.assertEqual(len(decoder_starts), 32)
            self.assertTrue(all("scheme_summary" not in event["prompt"] for event in decoder_starts))
            self.assertTrue((run_dir / "artifact_checksums.json").is_file())


if __name__ == "__main__":
    unittest.main()
