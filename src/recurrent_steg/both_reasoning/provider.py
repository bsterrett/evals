"""Anthropic encoder that thinks with adaptive thinking before writing its paraphrase."""
from __future__ import annotations

import os
from typing import Any

from eval_core.models import GenerationRequest
from eval_core.providers.anthropic import AnthropicProvider
from eval_core.providers.base import ProviderError
from recurrent_steg.reasoning_decoder.provider import EFFORTS, REQUEST_TIMEOUT_SECONDS


class ThinkingAnthropicProvider(AnthropicProvider):
    """Plain-text reply after adaptive thinking; the reasoning decoder's settings without its answer schema."""

    def __init__(self, config: Any, *, client: Any | None = None) -> None:
        if config.thinking_type != "adaptive" or config.reasoning_effort not in EFFORTS:
            raise ProviderError("The reasoning encoder needs adaptive thinking and an explicit effort level")
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProviderError("ANTHROPIC_API_KEY is not configured")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ProviderError("The Anthropic SDK is not installed; run `uv sync --extra providers`") from exc
            # A timed-out call leaves an uncertain charge that blocks the campaign ledger.
            client = Anthropic(max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS)
        super().__init__(config, client=client)

    def _input_args(self, request: GenerationRequest) -> dict[str, Any]:
        args = super()._input_args(request)
        # Summaries let the encoder's chosen convention be inspected; billing covers the full thinking.
        args["thinking"] = {"type": "adaptive", "display": "summarized"}
        args["output_config"] = {"effort": self.config.reasoning_effort}
        return args


def thinking_tokens(raw: dict | None) -> int | None:
    """Thinking tokens the API reported for a response, or None when the field is absent."""
    usage = (raw or {}).get("usage") or {}
    value = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    return value if type(value) is int and value >= 0 else None
