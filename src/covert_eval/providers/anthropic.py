from __future__ import annotations

import os
from typing import Any

from covert_eval.models import GenerationRequest, GenerationResponse, TokenUsage
from covert_eval.providers.base import Provider, ProviderError


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return {"repr": repr(obj)}


class AnthropicProvider(Provider):
    def __init__(self, config: Any, *, client: Any | None = None) -> None:
        super().__init__(config)
        if client is not None:
            self.client = client
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not configured")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderError(
                "The Anthropic SDK is not installed; run `uv sync --extra providers`"
            ) from exc
        self.client = Anthropic(max_retries=0, timeout=90.0)

    def _input_args(self, request: GenerationRequest) -> dict[str, Any]:
        args: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if self.config.thinking_type and self.config.thinking_type != "omit":
            args["thinking"] = {"type": self.config.thinking_type}
        return args

    def count_input_tokens(self, request: GenerationRequest) -> int:
        try:
            result = self.client.messages.count_tokens(**self._input_args(request))
        except Exception as exc:
            raise ProviderError(f"Anthropic token count failed: {exc}") from exc
        count = _value(result, "input_tokens")
        if type(count) is not int or count < 0:
            raise ProviderError("Anthropic token count response was invalid")
        return count

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        args = self._input_args(request)
        args["max_tokens"] = request.max_output_tokens
        try:
            response = self.client.messages.create(**args)
        except Exception as exc:
            raise ProviderError(f"Anthropic generation failed: {exc}") from exc
        raw = _dump(response)
        errors: list[str] = []
        content = _value(response, "content", [])
        text_parts = [
            _value(block, "text", "")
            for block in content
            if _value(block, "type") == "text" and _value(block, "text", "")
        ]
        text = "\n".join(text_parts).strip()
        if not text:
            errors.append("response did not include nonempty text content")
        stop_reason = _value(response, "stop_reason")
        stop_reason_text = str(stop_reason) if stop_reason is not None else "unknown"
        stop_details = _value(response, "stop_details")
        if stop_reason_text != "end_turn":
            errors.append(f"response stop_reason was {stop_reason_text!r}")
        if stop_details is not None:
            errors.append("response included stop_details")
        usage = _value(response, "usage")
        token_usage: TokenUsage | None = None
        if usage is None:
            errors.append("response did not include usage")
        else:
            ordinary = _value(usage, "input_tokens")
            output = _value(usage, "output_tokens")
            cached = _value(usage, "cache_read_input_tokens", 0)
            cache_write = _value(usage, "cache_creation_input_tokens", 0)
            if not all(
                type(value) is int and value >= 0
                for value in (ordinary, output, cached, cache_write)
            ):
                errors.append("response included missing or invalid token usage")
            else:
                token_usage = TokenUsage(
                    input_tokens=ordinary,
                    output_tokens=output,
                    cached_input_tokens=cached,
                    cache_write_input_tokens=cache_write,
                )
        return GenerationResponse(
            text=text,
            model=str(_value(response, "model", "")),
            usage=token_usage,
            response_id=_value(response, "id"),
            status="completed" if not errors else "rejected",
            stop_reason=stop_reason_text,
            raw=raw,
            operational_error="; ".join(errors) if errors else None,
        )

