"""Anthropic provider that constrains the reply to one of the request's answer choices."""
from __future__ import annotations

import json
from typing import Any

from eval_core.models import GenerationRequest
from eval_core.providers.anthropic import AnthropicProvider
from eval_core.providers.base import ProviderError


def answer_format(choices: list[str] | tuple[str, ...]) -> dict:
    """Structured-output schema allowing exactly {"answer": <choice>}, choices in the order shown."""
    return {"type": "json_schema", "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": list(choices)}},
        "required": ["answer"],
        "additionalProperties": False,
    }}


def parse_answer(text: str | None, choices: list[str] | tuple[str, ...]) -> dict:
    """Same output fields as the frozen decoder parser: answer, bare, final_text."""
    answer = None
    try:
        value = json.loads(text or "")
        if isinstance(value, dict) and set(value) == {"answer"} and value["answer"] in choices:
            answer = value["answer"]
    except ValueError:
        pass
    return dict(answer=answer, bare=answer is not None, final_text=text)


class AnswerOnlyAnthropicProvider(AnthropicProvider):
    def _input_args(self, request: GenerationRequest) -> dict[str, Any]:
        args = super()._input_args(request)
        choices = request.metadata.get("answer_choices")
        if not choices or len(choices) != 2 or len(set(choices)) != 2:
            raise ProviderError("Answer-only decoding needs exactly two distinct answer_choices")
        args["output_config"] = {"format": answer_format(choices)}
        return args
