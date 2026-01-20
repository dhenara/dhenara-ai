"""Service tests for AIModelClient retry behaviour."""

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types.genai.dhenara.request import AIModelCallConfig


class _FailingProvider:
    """Provider stub that always raises TimeoutError to drive retry logic."""

    def __init__(self) -> None:
        self.call_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def format_and_generate_response_sync(self, *args, **kwargs):
        self.call_count += 1
        raise TimeoutError("simulated failure")


@pytest.mark.service
@pytest.mark.case_id("DAI-029")
def test_retries_with_exponential_backoff_and_fail(monkeypatch, text_endpoint):
    """
    GIVEN a provider that repeatedly times out
    WHEN generate() is invoked with retries configured
    THEN the client should retry the configured number of times before surfacing the error
    """

    provider = _FailingProvider()

    def fake_provider(cls, model_endpoint, config, is_async):
        assert not is_async
        return provider

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=AIModelCallConfig(retries=3, retry_delay=0.0, max_retry_delay=0.0, timeout=None),
        is_async=False,
    )

    with pytest.raises(TimeoutError):
        client.generate(prompt="retry")

    assert provider.call_count == 3
