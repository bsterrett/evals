from eval_core.providers import Provider, ProviderError, ProviderRefusal
from eval_core.providers import create_provider as _create_provider


def create_provider(config, *, dry_run):
    from covert_eval.providers.mock import MockProvider
    return _create_provider(config, dry_run=dry_run, mock_factory=MockProvider)


__all__ = ["Provider", "ProviderError", "ProviderRefusal", "create_provider"]
