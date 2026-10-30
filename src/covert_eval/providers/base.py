from __future__ import annotations

from abc import ABC, abstractmethod

from covert_eval.models import GenerationRequest, GenerationResponse, ModelConfig


class ProviderError(RuntimeError):
    pass


class ProviderRefusal(ProviderError):
    """A terminal provider refusal with verified model identity and known usage."""

    def __init__(
        self,
        message: str,
        *,
        category: str | None = None,
        stop_details: object | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.stop_details = stop_details


class Provider(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def count_input_tokens(self, request: GenerationRequest) -> int:
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError


def create_provider(config: ModelConfig, *, dry_run: bool) -> Provider:
    if dry_run or config.provider == "mock":
        from covert_eval.providers.mock import MockProvider

        return MockProvider(config)
    if config.provider == "openai":
        from covert_eval.providers.openai import OpenAIProvider

        return OpenAIProvider(config)
    if config.provider == "anthropic":
        from covert_eval.providers.anthropic import AnthropicProvider

        return AnthropicProvider(config)
    raise ProviderError(f"Unsupported provider: {config.provider}")

