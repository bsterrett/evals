"""Anthropic decoder that reasons with adaptive thinking before a schema-enforced two-option answer."""
from __future__ import annotations

import os
from typing import Any

from eval_core.models import GenerationRequest
from eval_core.providers.base import ProviderError
from recurrent_steg.reference_amendment.provider import AnswerOnlyAnthropicProvider

# Thinking calls run far longer than bare answers; a client timeout would leave the
# charge uncertain and block the campaign ledger, so allow ten minutes per call.
REQUEST_TIMEOUT_SECONDS = 600.0
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ReasoningAnswerAnthropicProvider(AnswerOnlyAnthropicProvider):
    """Same prompt and answer schema as the answer-only decoder; only the thinking setting differs."""

    def __init__(self, config: Any, *, client: Any | None = None) -> None:
        if config.thinking_type != "adaptive" or config.reasoning_effort not in EFFORTS:
            raise ProviderError("The reasoning decoder needs adaptive thinking and an explicit effort level")
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProviderError("ANTHROPIC_API_KEY is not configured")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ProviderError("The Anthropic SDK is not installed; run `uv sync --extra providers`") from exc
            client = Anthropic(max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS)
        super().__init__(config, client=client)

    def _input_args(self, request: GenerationRequest) -> dict[str, Any]:
        args = super()._input_args(request)
        # Summaries let the decoder's stated convention be inspected; billing covers the full thinking.
        args["thinking"] = {"type": "adaptive", "display": "summarized"}
        args["output_config"] = dict(args["output_config"], effort=self.config.reasoning_effort)
        return args


def thinking_summary(raw: dict | None) -> dict:
    """Thinking blocks from a saved response: summaries, block count, and redacted blocks."""
    blocks = (raw or {}).get("content") or []
    texts = [block.get("thinking") or "" for block in blocks if isinstance(block, dict) and block.get("type") == "thinking"]
    redacted = sum(1 for block in blocks if isinstance(block, dict) and block.get("type") == "redacted_thinking")
    return dict(blocks=len(texts) + redacted, redacted=redacted, summary="\n\n".join(text for text in texts if text))
