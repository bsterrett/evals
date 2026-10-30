from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

from covert_eval.budget import BudgetLedger
from covert_eval.config import load_config
from covert_eval.experiment import ExperimentRunner, _write_artifact_checksums, _write_json
from covert_eval.models import GenerationRequest, GenerationResponse, TokenUsage
from covert_eval.providers.base import Provider, ProviderError, ProviderRefusal


ROOT = Path(__file__).resolve().parents[3]
TEST_TMP = ROOT / ".test-tmp"


class _FixtureProvider(Provider):
    def __init__(self, config: object, response: GenerationResponse) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.response = response

    def count_input_tokens(self, request: GenerationRequest) -> int:
        return 10

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return self.response


class FailureAccountingTests(unittest.TestCase):
    def _runner(self, directory: Path) -> tuple[ExperimentRunner, object]:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
        config = replace(
            config,
            experiment=replace(config.experiment, output_dir=directory / "runs"),
            budget=replace(config.budget, ledger=directory / "ledger.json"),
            subjects=tuple(replace(subject, rate=replace(subject.rate, verified_at=date.today())) for subject in config.subjects),
            judge=replace(config.judge, rate=replace(config.judge.rate, verified_at=date.today())),
        )
        subject = config.subjects[1]
        runner = ExperimentRunner(config, dry_run=False, selected_subjects={subject.id}, case_limit=1)
        runner.run_dir.mkdir(parents=True)
        runner.ledger = BudgetLedger(config.budget.ledger, config.budget.campaign_usd)
        return runner, subject

    def test_incomplete_response_is_logged_settled_and_blocks(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            runner, subject = self._runner(Path(directory))
            response = GenerationResponse(
                text="partial",
                model=subject.model,  # type: ignore[attr-defined]
                usage=TokenUsage(input_tokens=10, output_tokens=20),
                response_id="resp-incomplete",
                status="incomplete",
                stop_reason="max_output_tokens",
                raw={"id": "resp-incomplete", "status": "incomplete"},
                operational_error="response status was 'incomplete'",
            )
            key = f"{subject.provider}:{subject.id}"  # type: ignore[attr-defined]
            runner.providers[key] = _FixtureProvider(subject, response)
            with self.assertRaises(ProviderError):
                runner._call(
                    subject,  # type: ignore[arg-type]
                    GenerationRequest(stage="encoder", prompt="test", max_output_tokens=30),
                    trial_id="fixture",
                )
            self.assertTrue(runner._budget().blocked)
            self.assertGreater(runner._budget().spent, 0)
            self.assertEqual(runner._budget().uncertain, 0)
            with runner.events_path.open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertIn("call_received", [event["event"] for event in events])
            received = next(event for event in events if event["event"] == "call_received")
            self.assertEqual(received["raw_response"]["id"], "resp-incomplete")
            self.assertIn("call_rejected", [event["event"] for event in events])

    def test_returned_model_mismatch_is_logged_and_uncertain(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            runner, subject = self._runner(Path(directory))
            response = GenerationResponse(
                text="answer",
                model="unexpected-model",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                response_id="resp-mismatch",
                status="completed",
                stop_reason="completed",
                raw={"id": "resp-mismatch", "model": "unexpected-model"},
            )
            key = f"{subject.provider}:{subject.id}"  # type: ignore[attr-defined]
            runner.providers[key] = _FixtureProvider(subject, response)
            with self.assertRaises(ProviderError):
                runner._call(
                    subject,  # type: ignore[arg-type]
                    GenerationRequest(stage="encoder", prompt="test", max_output_tokens=30),
                    trial_id="fixture",
                )
            self.assertTrue(runner._budget().blocked)
            self.assertGreater(runner._budget().uncertain, 0)
            with runner.events_path.open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            received = next(event for event in events if event["event"] == "call_received")
            self.assertEqual(received["returned_model"], "unexpected-model")

    def test_refusal_is_settled_without_blocking(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            runner, subject = self._runner(Path(directory))
            response = GenerationResponse(
                text="",
                model=subject.model,  # type: ignore[attr-defined]
                usage=TokenUsage(input_tokens=10, output_tokens=0),
                response_id="resp-refusal",
                status="completed",
                stop_reason="refusal",
                raw={
                    "id": "resp-refusal",
                    "stop_reason": "refusal",
                    "stop_details": {
                        "type": "refusal",
                        "category": "reasoning_extraction",
                    },
                },
                operational_error="provider refusal",
            )
            key = f"{subject.provider}:{subject.id}"  # type: ignore[attr-defined]
            runner.providers[key] = _FixtureProvider(subject, response)
            with self.assertRaises(ProviderRefusal) as raised:
                runner._call(
                    subject,  # type: ignore[arg-type]
                    GenerationRequest(stage="control", prompt="test", max_output_tokens=30),
                    trial_id="fixture",
                )
            self.assertEqual(raised.exception.category, "reasoning_extraction")
            self.assertFalse(runner._budget().blocked)
            self.assertGreater(runner._budget().spent, 0)
            self.assertEqual(runner._budget().uncertain, 0)
            self.assertGreater(runner.run_spent_usd, 0)
            with runner.events_path.open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            rejected = next(event for event in events if event["event"] == "call_rejected")
            self.assertEqual(rejected["budget_status"], "settled_refusal")

    def test_trial_refusals_are_recorded_and_do_not_abort_run(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
            config = replace(
                config,
                experiment=replace(
                    config.experiment,
                    output_dir=Path(directory) / "runs",
                    max_cases=2,
                ),
                budget=replace(config.budget, ledger=Path(directory) / "ledger.json"),
            )
            subject = config.subjects[1]
            runner = ExperimentRunner(
                config,
                dry_run=True,
                selected_subjects={subject.id},
                case_limit=2,
            )
            with patch.object(
                runner,
                "_run_trial",
                side_effect=[ProviderRefusal("first"), ProviderRefusal("second")],
            ):
                run_dir = runner.run()
            with (run_dir / "results.jsonl").open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["failure_class"] == "provider_refusal" for row in rows))
            with (run_dir / "manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["operationally_completed_trials"], 0)

    def test_resume_inherits_attempted_trials_without_rerunning_them(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
            config = replace(
                config,
                experiment=replace(config.experiment, output_dir=root / "runs"),
                budget=replace(config.budget, ledger=root / "ledger.json"),
            )
            subject = config.subjects[1]
            source_runner = ExperimentRunner(
                config,
                dry_run=True,
                selected_subjects={subject.id},
                case_limit=1,
            )
            source_dir = source_runner.run()
            with (source_dir / "manifest.json").open("r", encoding="utf-8") as handle:
                source_manifest = json.load(handle)
            source_manifest["status"] = "failed"
            _write_json(source_dir / "manifest.json", source_manifest)
            _write_artifact_checksums(source_dir)

            resumed_runner = ExperimentRunner(
                config,
                dry_run=True,
                selected_subjects={subject.id},
                case_limit=1,
                resume_from=source_dir,
            )
            with patch.object(
                resumed_runner,
                "_run_trial",
                side_effect=AssertionError("inherited trial was rerun"),
            ):
                resumed_dir = resumed_runner.run()
            with (resumed_dir / "results.jsonl").open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            with (resumed_dir / "manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["resumed_from"]["inherited_trials"], 1)


if __name__ == "__main__":
    unittest.main()
