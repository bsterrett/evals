from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from covert_eval.config import load_config
from covert_eval.experiment import ExperimentRunner


ROOT = Path(__file__).resolve().parents[3]
TEST_TMP = ROOT / ".test-tmp"


class ExperimentTests(unittest.TestCase):
    def test_end_to_end_dry_run(self) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            temp = Path(directory)
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=temp / "runs", max_cases=1),
                budget=replace(config.budget, ledger=temp / "ledger.json"),
            )
            subject_id = "openai_gpt_6_luna_no_reasoning"
            runner = ExperimentRunner(
                config,
                dry_run=True,
                selected_subjects={subject_id},
                case_limit=1,
            )
            run_dir = runner.run()
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "events.jsonl").is_file())
            self.assertTrue((run_dir / "results.jsonl").is_file())
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["total_trials"], 1)
            self.assertEqual(summary["outcome_schema_version"], 2)
            self.assertEqual(
                summary["subjects"][subject_id]["v1_pair_success_all_attempts"]["successes"],
                0,
            )
            self.assertTrue((run_dir / "artifact_checksums.json").is_file())
            with (run_dir / "results.jsonl").open("r", encoding="utf-8") as handle:
                result = json.loads(next(handle))
            self.assertEqual(result["outcome_schema_version"], 2)
            self.assertIn(result["operational_status"], {"completed"})
            self.assertIn(
                result["encoder_behavior"],
                {"candidate_passage", "textual_refusal", "off_task", "malformed"},
            )
            self.assertIn(
                result["primary_outcome"],
                {
                    "success",
                    "relaxed_transfer",
                    "failed_recovery",
                    "failed_cover",
                    "failed_control",
                    "behavioral_refusal",
                    "evaluation_failure",
                },
            )
            with (run_dir / "manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(manifest["outcome_schema_version"], 2)
            with (run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(sum(event["event"] == "call_completed" for event in events), 7)
            recovery_starts = [
                event
                for event in events
                if event["event"] == "call_started" and event["stage"] == "judge_recovery"
            ]
            self.assertEqual(len(recovery_starts), 1)
            self.assertNotIn("Dry-run provider: no semantic inference", recovery_starts[0]["prompt"])
            self.assertIn("No recoverable message was identified.", recovery_starts[0]["prompt"])
            self.assertNotIn("is_recoverable", recovery_starts[0]["prompt"])
            cover_starts = [
                event
                for event in events
                if event["event"] == "call_started" and event["stage"] == "judge_cover"
            ]
            self.assertEqual(len(cover_starts), 1)
            with (run_dir / "cases.jsonl").open("r", encoding="utf-8") as handle:
                selected_case = json.loads(next(handle))
            self.assertNotIn(selected_case["id"], config.experiment.excluded_case_ids)
            self.assertNotIn(selected_case["implicit_message"], cover_starts[0]["prompt"])
            self.assertNotIn("implicit_message_verbatim", cover_starts[0]["prompt"])


if __name__ == "__main__":
    unittest.main()

