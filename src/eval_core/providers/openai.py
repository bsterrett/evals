from __future__ import annotations

import os
from typing import Any

from eval_core.models import GenerationRequest, GenerationResponse, TokenUsage
from eval_core.providers.base import Provider, ProviderError


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


class OpenAIProvider(Provider):
    def __init__(self, config: Any, *, client: Any | None = None) -> None:
        super().__init__(config)
        if client is not None:
            self.client = client
            return
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError(
                "The OpenAI SDK is not installed; run `uv sync --extra providers`"
            ) from exc
        self.client = OpenAI(max_retries=0, timeout=90.0)

    def _input_args(self, request: GenerationRequest) -> dict[str, Any]:
        args: dict[str, Any] = {
            "model": self.config.model,
            "input": request.prompt,
        }
        if self.config.reasoning_effort:
            args["reasoning"] = {"effort": self.config.reasoning_effort}
        return args

    def count_input_tokens(self, request: GenerationRequest) -> int:
        try:
            result = self.client.responses.input_tokens.count(**self._input_args(request))
        except Exception as exc:
            raise ProviderError(f"OpenAI token count failed: {exc}") from exc
        count = _value(result, "input_tokens")
        if type(count) is not int or count < 0:
            raise ProviderError("OpenAI token count response was invalid")
        return count

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        args = self._input_args(request)
        args.update(
            {
                "max_output_tokens": request.max_output_tokens,
                "service_tier": "default",
                "store": False,
            }
        )
        try:
            response = self.client.responses.create(**args)
        except Exception as exc:
            raise ProviderError(f"OpenAI generation failed: {exc}") from exc
        raw = _dump(response)
        status = _value(response, "status")
        status_text = str(status) if status is not None else "unknown"
        errors: list[str] = []
        if status_text != "completed":
            errors.append(f"response status was {status_text!r}")
        response_error = _value(response, "error")
        if response_error is not None:
            errors.append("response included an error object")
        usage = _value(response, "usage")
        token_usage: TokenUsage | None = None
        if usage is None:
            errors.append("response did not include usage")
        else:
            total_input = _value(usage, "input_tokens")
            output = _value(usage, "output_tokens")
            details = _value(usage, "input_tokens_details", {})
            cached = _value(details, "cached_tokens", 0)
            cache_write = _value(details, "cache_write_tokens", 0)
            if not all(
                type(value) is int and value >= 0
                for value in (total_input, output, cached, cache_write)
            ):
                errors.append("response included invalid token usage")
            else:
                ordinary = total_input - cached - cache_write
                if ordinary < 0:
                    errors.append("input token buckets did not sum correctly")
                else:
                    token_usage = TokenUsage(
                        input_tokens=ordinary,
                        output_tokens=output,
                        cached_input_tokens=cached,
                        cache_write_input_tokens=cache_write,
                    )
        text = _value(response, "output_text", "")
        if not isinstance(text, str):
            text = ""
            errors.append("response output_text was not a string")
        elif not text.strip():
            errors.append("response did not include nonempty output text")
        incomplete_details = _value(response, "incomplete_details")
        incomplete_reason = _value(incomplete_details, "reason") if incomplete_details else None
        return GenerationResponse(
            text=text.strip(),
            model=str(_value(response, "model", "")),
            usage=token_usage,
            response_id=_value(response, "id"),
            status=status_text,
            stop_reason=str(incomplete_reason) if incomplete_reason else status_text,
            raw=raw,
            operational_error="; ".join(errors) if errors else None,
        )
