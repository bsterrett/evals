from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from eval_core.budget import BudgetExceeded, BudgetLedger
from eval_core.models import BudgetSettings, GenerationRequest, GenerationResponse, ModelConfig, RateCard, TokenUsage
from eval_core.providers import Provider
from eval_core.runtime import RunSession
from eval_core.workspace import find_project_root

ROOT = Path(__file__).resolve().parents[2]


class OverlappingProvider(Provider):
    def __init__(self, config, barrier):
        super().__init__(config)
        self.barrier = barrier

    def count_input_tokens(self, request):
        return 10

    def generate(self, request):
        self.barrier.wait(timeout=10)
        return GenerationResponse("Paris", self.config.model, TokenUsage(10, 1), None, "completed", "completed", {})


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)

    def test_core_imports_without_loading_the_existing_experiment(self):
        subprocess.run([
            sys.executable, "-B", "-c",
            "import sys; from eval_core.runtime import RunSession; assert not any(k.startswith('covert_eval') for k in sys.modules)",
        ], check=True, cwd=ROOT)

    def test_unrelated_experiments_overlap_calls_and_share_campaign(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
            root = Path(directory)
            settings = BudgetSettings(Decimal("1"), root / "campaign.sqlite3", 30)
            rate = RateCard(Decimal("1000"), Decimal("1000"), 100, date.today(), "https://example.com/prices")
            model = ModelConfig("fixture", "mock", "fixture", rate)
            barrier = threading.Barrier(2)
            factory = lambda config, **kwargs: OverlappingProvider(config, barrier)

            def run(experiment_id):
                session = RunSession(experiment_id=experiment_id, output_dir=root / experiment_id,
                                     budget=settings, dry_run=False, run_budget_usd=Decimal(".1"), provider_factory=factory)
                with session.accounting():
                    response, _ = session.call(model, GenerationRequest("capital_question", "Capital of France?", 20), trial_id="one")
                return session, response

            with ThreadPoolExecutor(max_workers=2) as executor:
                first, second = list(executor.map(run, ("geography", "answer_format")))
            self.assertEqual(first[1].text, "Paris")
            self.assertEqual(second[1].text, "Paris")
            self.assertNotEqual(first[0].events_path, second[0].events_path)
            ledger = BudgetLedger(settings.ledger, settings.campaign_usd)
            self.assertEqual(ledger.spent, Decimal(".022"))
            self.assertEqual(ledger.pending, 0)
            self.assertEqual({row["experiment_id"] for row in ledger.export()["calls"]}, {"geography", "answer_format"})

    def test_nested_config_discovers_actual_source_root(self):
        config = ROOT / "experiments/semantic_covert_channel/configs/v1.toml"
        self.assertEqual(find_project_root(config), ROOT)

    def test_run_ceiling_and_stale_prices_stop_before_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
            root = Path(directory)
            settings = BudgetSettings(Decimal("1"), root / "campaign.sqlite3", 30)
            rate = RateCard(Decimal("1000"), Decimal("1000"), 100, date.today(), "https://example.com/prices")
            model = ModelConfig("fixture", "mock", "fixture", rate)
            # No provider may reach generate(): the barrier would otherwise time out.
            factory = lambda config, **kwargs: OverlappingProvider(config, threading.Barrier(2))
            session = RunSession(experiment_id="cap", output_dir=root, budget=settings, dry_run=False,
                                 run_budget_usd=Decimal(".001"), provider_factory=factory)
            with session.accounting():
                with self.assertRaises(BudgetExceeded):
                    session.call(model, GenerationRequest("question", "test", 20), trial_id="one")
                self.assertEqual(session.budget().pending, 0)
            from dataclasses import replace
            model = replace(model, rate=replace(rate, verified_at=date.today() - timedelta(days=31)))
            with session.accounting():
                with self.assertRaisesRegex(ValueError, "Pricing"):
                    session.call(model, GenerationRequest("question", "test", 20), trial_id="two")
                self.assertEqual(session.budget().spent, 0)
