from __future__ import annotations

import unittest
from types import SimpleNamespace

from covert_eval.config import load_config
from eval_core.models import GenerationRequest
from eval_core.providers.anthropic import AnthropicProvider
from eval_core.providers.openai import OpenAIProvider

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class _CreateEndpoint:
    def __init__(self, response: object) -> None:
        self.response = response

    def create(self, **_: object) -> object:
        return self.response


class ProviderFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "experiments" / "semantic_covert_channel" / "configs" / "v1.toml")
        cls.openai_config = config.subjects[1]
        cls.anthropic_config = config.subjects[2]
        cls.request = GenerationRequest(stage="encoder", prompt="test", max_output_tokens=20)

    def test_openai_incomplete_preserves_usage_and_raw_response(self) -> None:
        response = {
            "id": "resp-test",
            "model": self.openai_config.model,
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "error": None,
            "output_text": "partial",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            },
        }
        client = SimpleNamespace(responses=_CreateEndpoint(response))
        result = OpenAIProvider(self.openai_config, client=client).generate(self.request)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.stop_reason, "max_output_tokens")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.raw["id"], "resp-test")
        self.assertIsNotNone(result.operational_error)

    def test_anthropic_missing_required_usage_is_rejected_not_zero(self) -> None:
        response = {
            "id": "msg-test",
            "model": self.anthropic_config.model,
            "stop_reason": "end_turn",
            "stop_details": None,
            "content": [{"type": "text", "text": "answer"}],
            "usage": {},
        }
        client = SimpleNamespace(messages=_CreateEndpoint(response))
        result = AnthropicProvider(self.anthropic_config, client=client).generate(self.request)
        self.assertIsNone(result.usage)
        self.assertIsNotNone(result.operational_error)

    def test_anthropic_max_tokens_is_operational_failure_with_usage(self) -> None:
        response = {
            "id": "msg-test",
            "model": self.anthropic_config.model,
            "stop_reason": "max_tokens",
            "stop_details": None,
            "content": [{"type": "text", "text": "partial"}],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 20,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
        client = SimpleNamespace(messages=_CreateEndpoint(response))
        result = AnthropicProvider(self.anthropic_config, client=client).generate(self.request)
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.status, "rejected")
        self.assertIn("max_tokens", result.operational_error or "")


if __name__ == "__main__":
    unittest.main()
