from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from covert_eval.calibration import (
    LunaDecoderCalibrationRunner,
    MultiModelDecoderContractRunner,
    RepeatedDecoderContractRunner,
    Stage1CDecoderContractRunner,
    _parse_decoder_with_schema_diagnostics,
    calibration_plan,
    decoder_contract_plan,
    stage1b_plan,
    stage1c_plan,
    summarize_decoder_contract,
)
from covert_eval.config import load_config


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / ".test-tmp"


class CalibrationTests(unittest.TestCase):
    def test_stage1c_plan_and_dry_run(self) -> None:
        config = load_config(ROOT / "configs" / "v1.toml")
        plan = stage1c_plan(config)
        self.assertEqual(plan["decoder_prompt_version"], "v1.4.0")
        self.assertEqual(plan["repetitions_per_subject"], 1)
        self.assertEqual(plan["decoder_calls"], 12)
        self.assertEqual(plan["judge_calls"], 4)
        self.assertEqual(plan["total_calls"], 16)
        self.assertLess(Decimal(plan["structural_preflight_bound_usd"]), Decimal("1.00"))

        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            temp = Path(directory)
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=temp / "runs"),
                budget=replace(config.budget, ledger=temp / "ledger.json"),
            )
            run_dir = Stage1CDecoderContractRunner(config, dry_run=True).run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["subjects_total"], 4)
            self.assertEqual(summary["decoder_prompt_version"], "v1.4.0")
            self.assertEqual(summary["repetitions_per_subject"], 1)
            with (run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(sum(event["event"] == "call_completed" for event in events), 16)

    def test_stage1b_scores_semantics_separately_from_exact_schema(self) -> None:
        parsed = _parse_decoder_with_schema_diagnostics(
            '{"best_separate_message_candidate":"The gate is blue.",'
            '"candidate_confidence":0.8,"is_recoverable":true,'
            '"evidence":"The passage states it.","extra_note":"unrequested"}'
        )
        self.assertTrue(parsed["semantic_valid"])
        self.assertFalse(parsed["exact_schema_valid"])
        self.assertEqual(parsed["unexpected_keys"], ["extra_note"])
        self.assertEqual(
            parsed["answer"]["best_separate_message_candidate"], "The gate is blue."
        )

    def test_stage1b_plan_and_dry_run(self) -> None:
        config = load_config(ROOT / "configs" / "v1.toml")
        plan = stage1b_plan(config)
        self.assertEqual(plan["repetitions_per_subject"], 3)
        self.assertEqual(plan["decoder_calls"], 36)
        self.assertEqual(plan["judge_calls"], 12)
        self.assertEqual(plan["total_calls"], 48)
        self.assertLess(Decimal(plan["structural_preflight_bound_usd"]), Decimal("1.00"))

        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            temp = Path(directory)
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=temp / "runs"),
                budget=replace(config.budget, ledger=temp / "ledger.json"),
            )
            run_dir = RepeatedDecoderContractRunner(config, dry_run=True).run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["subjects_total"], 4)
            self.assertEqual(summary["decoder_prompt_version"], "v1.3.0")
            self.assertTrue(
                all(
                    subject["completed_repetitions"] == 3
                    for subject in summary["subjects"].values()
                )
            )
            with (run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(sum(event["event"] == "call_completed" for event in events), 48)

    def test_decoder_contract_partial_summary_preserves_invalid_decoder(self) -> None:
        invalid = {"valid": False, "answer": None, "error": "extra key"}
        valid = {
            "valid": True,
            "answer": {
                "best_separate_message_candidate": "The gate is blue.",
                "candidate_confidence": 0.8,
                "is_recoverable": True,
                "evidence": "A separate detail says so.",
            },
            "error": None,
        }
        summary = summarize_decoder_contract(
            {
                "subject": {
                    "model": "example-model",
                    "provider": "example-provider",
                    "contract_pass": False,
                    "decoders": {
                        "clean": invalid,
                        "anomalous": valid,
                        "unambiguous": valid,
                    },
                    "scores": {
                        "clean": {"payload_recovery": 0},
                        "anomalous": {"payload_recovery": 3},
                        "unambiguous": {"payload_recovery": 4},
                    },
                }
            },
            dry_run=False,
        )
        self.assertFalse(summary["subjects"]["subject"]["clean"]["valid"])
        self.assertEqual(summary["subjects"]["subject"]["clean"]["error"], "extra key")
        self.assertIsNone(summary["subjects"]["subject"]["clean"]["candidate"])

    def test_stage1_decoder_contract_plan_and_dry_run(self) -> None:
        config = load_config(ROOT / "configs" / "v1.toml")
        plan = decoder_contract_plan(config)
        self.assertEqual(plan["decoder_calls"], 12)
        self.assertEqual(plan["judge_calls"], 4)
        self.assertEqual(plan["total_calls"], 16)
        self.assertLess(Decimal(plan["structural_preflight_bound_usd"]), Decimal("1.00"))

        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            temp = Path(directory)
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=temp / "runs"),
                budget=replace(config.budget, ledger=temp / "ledger.json"),
            )
            run_dir = MultiModelDecoderContractRunner(config, dry_run=True).run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["subjects_total"], 4)
            self.assertEqual(summary["decoder_prompt_version"], "v1.2.0")
            with (run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(sum(event["event"] == "call_completed" for event in events), 16)

    def test_plan_and_complete_dry_calibration(self) -> None:
        config = load_config(ROOT / "configs" / "v1.toml")
        plan = calibration_plan(config)
        self.assertEqual(plan["total_calls"], 16)
        self.assertLess(Decimal(plan["structural_preflight_bound_usd"]), Decimal("1.00"))

        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            temp = Path(directory)
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=temp / "runs"),
                budget=replace(config.budget, ledger=temp / "ledger.json"),
            )
            run_dir = LunaDecoderCalibrationRunner(config, dry_run=True).run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(set(summary["protocols"]), {
                "current_nullable",
                "current_forced",
                "clarified_nullable",
                "clarified_forced",
            })
            with (run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(sum(event["event"] == "call_completed" for event in events), 16)
            self.assertTrue((run_dir / "artifact_checksums.json").is_file())


if __name__ == "__main__":
    unittest.main()
