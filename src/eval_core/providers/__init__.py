from collections.abc import Callable

from eval_core.models import GenerationRequest, GenerationResponse, ModelConfig, TokenUsage
from eval_core.providers.base import Provider, ProviderError, ProviderRefusal


class DryRunProvider(Provider):
    """Generic structural placeholder; experiments may supply their own mock."""

    def count_input_tokens(self, request: GenerationRequest) -> int:
        return max(1, len(request.prompt.encode("utf-8")) // 4)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="Dry run: no model call was made.", model=self.config.model,
            usage=TokenUsage(self.count_input_tokens(request), 0), response_id=None,
            status="completed", stop_reason="completed", raw={"dry_run": True},
        )


def create_provider(
    config: ModelConfig, *, dry_run: bool,
    mock_factory: Callable[[ModelConfig], Provider] = DryRunProvider,
) -> Provider:
    if dry_run or config.provider == "mock":
        return mock_factory(config)
    if config.provider == "openai":
        from eval_core.providers.openai import OpenAIProvider
        return OpenAIProvider(config)
    if config.provider == "anthropic":
        from eval_core.providers.anthropic import AnthropicProvider
        return AnthropicProvider(config)
    raise ProviderError(f"Unsupported provider: {config.provider}")


__all__ = ["Provider", "ProviderError", "ProviderRefusal", "create_provider"]
