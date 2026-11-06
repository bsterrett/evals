from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import unittest
from unittest.mock import patch

from eval_core.local.protocol import LocalRequest, identity, verify_observations
from eval_core.local.worker import make_server
from eval_core.models import BudgetSettings, GenerationRequest, ModelConfig, RateCard
from eval_core.providers import create_provider, ProviderError
from eval_core.providers.local import LocalProvider
from eval_core.runtime import RunSession
from eval_core.budget import BudgetLedger


class FixtureBackend:
    model_id = identity("ouro")
    depth = 4
    max_total = 64
    capabilities = dict(protocol_version=1, model=model_id, depth=depth, max_total_tokens=max_total)

    def __init__(self):
        self.calls = 0
        self.corrupt = False

    def tokenize(self, request):
        if len(request.prompt) + request.max_output_tokens > self.max_total:
            raise ValueError("Context exceeded")
        return list(range(len(request.prompt)))

    def generate(self, request):
        self.calls += 1
        return dict(protocol_version=1, model=self.model_id, depth=self.depth,
            seed=request.seed, prompt_format=request.prompt_format, text="OK",
            input_tokens=len(self.tokenize(request)), output_tokens=2, output_token_ids=[3, 4],
            observed_iterations=[4, 3 if self.corrupt else 4], stop_reason="max_output_tokens",
            response_id="fixture")


class LocalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.backend = FixtureBackend()
        self.server = make_server(self.backend, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.config = ModelConfig("ouro", "local", self.backend.model_id,
            RateCard(Decimal(0), Decimal(0), 63, date.today(), "local"))
        self.provider = LocalProvider(self.config, base_url=self.url)
        self.request = GenerationRequest("test", "Hi", 2,
            {"local": dict(depth=4, seed=42, prompt_format="raw-v1")})

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_http_roundtrip_and_real_accounting(self):
        self.assertEqual(self.provider.count_input_tokens(self.request), 2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = RunSession(experiment_id="fixture", output_dir=root / "runs", dry_run=False,
                budget=BudgetSettings(Decimal(1), root / "ledger.sqlite3", 1),
                provider_factory=lambda config, dry_run: self.provider)
            with session.accounting():
                response, cost = session.call(self.config, self.request, trial_id="one")
            self.assertEqual(response.text, "OK")
            self.assertEqual(cost, Decimal(0))
            events = [json.loads(line) for line in session.events_path.read_text().splitlines()]
            received = next(e for e in events if e["event"] == "call_received")
            self.assertEqual(received["raw_response"]["observed_iterations"], [4, 4])
            ledger = BudgetLedger(root / "ledger.sqlite3", Decimal(1))
            self.assertEqual(len(ledger.export()["calls"]), 1)
            self.assertEqual(ledger.spent, Decimal(0))
            self.assertEqual(ledger.pending, Decimal(0))
            with self.assertRaises(ValueError):
                ledger.reserve(Decimal("-0.01"), {})

    def test_observed_depth_mismatch_fails(self):
        self.backend.corrupt = True
        with self.assertRaisesRegex(ProviderError, "Observed recurrent"):
            self.provider.generate(self.request)

    def test_wrong_revision_and_depth_fail_before_inference(self):
        wrong = LocalProvider(replace(self.config, model="wrong@revision"), base_url=self.url)
        with self.assertRaisesRegex(ProviderError, "model/revision"):
            wrong.count_input_tokens(self.request)
        metadata = deepcopy(self.request.metadata)
        metadata["local"]["depth"] = 3
        with self.assertRaisesRegex(ProviderError, "does not match"):
            self.provider.generate(replace(self.request, metadata=metadata))
        self.assertEqual(self.backend.calls, 0)

    def test_context_overflow_fails_without_truncation(self):
        with self.assertRaisesRegex(ProviderError, "Context exceeded"):
            self.provider.count_input_tokens(replace(self.request, prompt="X" * 64))

    def test_unknown_controls_and_boolean_depth_are_rejected(self):
        value = dict(model=self.backend.model_id, depth=4, prompt="test", max_output_tokens=2,
            seed=42, prompt_format="raw-v1")
        with self.assertRaises(ValueError):
            LocalRequest.parse(dict(value, temperature=0.7), model=value["model"], depth=4, max_total=64)
        with self.assertRaises(ValueError):
            LocalRequest.parse(dict(value, depth=True), model=value["model"], depth=1, max_total=64)
        with self.assertRaises(ValueError):
            verify_observations([4], 4, 2)

    def test_loopback_only_and_dedicated_factory(self):
        with self.assertRaisesRegex(ProviderError, "loopback"):
            LocalProvider(self.config, base_url="http://192.0.2.1:8010")
        with patch.dict("os.environ", {"EVAL_LOCAL_URL": self.url, "OPENAI_BASE_URL": "https://unrelated.invalid"}):
            provider = create_provider(self.config, dry_run=False)
            self.assertIsInstance(provider, LocalProvider)
            self.assertEqual(provider.count_input_tokens(self.request), 2)

    def test_controller_imports_do_not_require_gpu_packages(self):
        subprocess.run([sys.executable, "-B", "-c",
            "import sys; from eval_core.local.backend import RecurrentBackend; "
            "from eval_core.providers.local import LocalProvider; "
            "from eval_core.local.dory_backend import DoryBackend; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], check=True)

    def test_nonzero_local_rates_are_rejected(self):
        rate = replace(self.config.rate, cached_input_usd_per_million=Decimal("1"))
        with self.assertRaisesRegex(ProviderError, "zero-charge"):
            LocalProvider(replace(self.config, rate=rate), base_url=self.url)


if __name__ == "__main__":
    unittest.main()
