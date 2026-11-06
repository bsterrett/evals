from __future__ import annotations

from dataclasses import asdict
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from eval_core.local.protocol import LocalRequest, PROTOCOL_VERSION, integer, verify_observations
from eval_core.models import GenerationRequest, GenerationResponse, ModelConfig, TokenUsage
from eval_core.providers.base import Provider, ProviderError


class LocalProvider(Provider):
    """Dedicated, revision-verified local transport, independent of OpenAI settings.

    request.metadata['local'] must specify depth, seed, and prompt_format.
    ModelConfig.model is the complete repo_id@revision from profiles.json.
    """

    def __init__(self, config: ModelConfig, *, base_url: str | None = None, timeout: float = 1800) -> None:
        super().__init__(config)
        self.base_url = (base_url or os.environ.get("EVAL_LOCAL_URL", "http://127.0.0.1:8010")).rstrip("/")
        url = urlparse(self.base_url)
        if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
                or url.username or url.password or url.path or url.query or url.fragment):
            raise ProviderError("EVAL_LOCAL_URL must be a loopback HTTP origin")
        if any((config.rate.input_usd_per_million, config.rate.output_usd_per_million,
                config.rate.cached_input_usd_per_million, config.rate.cache_write_usd_per_million)):
            raise ProviderError("Local inference must use a zero-charge rate card")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def _call(self, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base_url + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise ProviderError(f"Local worker HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError, ValueError) as exc:
            raise ProviderError(f"Local worker request failed: {exc}") from exc
        if not isinstance(result, dict) or result.get("protocol_version") != PROTOCOL_VERSION:
            raise ProviderError("Unsupported local worker protocol")
        if result.get("model") != self.config.model:
            raise ProviderError("Local worker model/revision does not match ModelConfig")
        return result

    def _payload(self, request: GenerationRequest) -> dict:
        settings = request.metadata.get("local")
        if not isinstance(settings, dict) or set(settings) != {"depth", "seed", "prompt_format"}:
            raise ProviderError("metadata.local must specify exactly depth, seed, and prompt_format")
        health = self._call("/health")
        if health.get("ready") is not True:
            raise ProviderError("Local worker is not ready")
        payload = dict(model=self.config.model, prompt=request.prompt,
            max_output_tokens=request.max_output_tokens, **settings)
        try:
            parsed = LocalRequest.parse(payload, model=self.config.model,
                depth=health["depth"], max_total=health["max_total_tokens"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError(f"Invalid local request: {exc}") from exc
        return asdict(parsed)

    def count_input_tokens(self, request: GenerationRequest) -> int:
        payload = self._payload(request)
        result = self._call("/tokenize", payload)
        try:
            if result["depth"] != payload["depth"] or result["prompt_format"] != payload["prompt_format"]:
                raise ValueError("Tokenization settings differ from request")
            count = integer(result["input_tokens"], "input_tokens", 1, self.config.rate.max_input_tokens)
            if len(result["input_token_ids"]) != count:
                raise ValueError("Token count differs from token IDs")
            return count
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError(f"Invalid local tokenization response: {exc}") from exc

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = self._payload(request)
        result = self._call("/generate", payload)
        try:
            for key in ("depth", "seed", "prompt_format"):
                if result[key] != payload[key]:
                    raise ValueError(f"Returned {key} differs from request")
            inputs = integer(result["input_tokens"], "input_tokens", 1, self.config.rate.max_input_tokens)
            outputs = integer(result["output_tokens"], "output_tokens", 1, request.max_output_tokens)
            verify_observations(result["observed_iterations"], payload["depth"], outputs)
            if len(result["output_token_ids"]) != outputs:
                raise ValueError("Output count differs from token IDs")
            if result["stop_reason"] not in ("eos", "max_output_tokens"):
                raise ValueError("Unknown stop reason")
            if result["stop_reason"] == "max_output_tokens" and outputs != request.max_output_tokens:
                raise ValueError("Output stopped before its declared token limit")
            if not isinstance(result["text"], str):
                raise ValueError("Output text must be a string")
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError(f"Invalid local generation response: {exc}") from exc
        return GenerationResponse(text=result["text"], model=result["model"],
            usage=TokenUsage(inputs, outputs), response_id=result["response_id"], status="completed",
            stop_reason=result["stop_reason"], raw=result)
