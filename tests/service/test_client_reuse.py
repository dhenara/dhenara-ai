"""Service tests for reusing existing AIModelClient connections."""

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types import ExternalApiCallStatusEnum

from ..helpers import FakeProvider


class _CountingProvider(FakeProvider):
    """Fake provider that tracks setup and cleanup counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_calls = 0
        self.cleanup_calls = 0

    def _setup_client_sync(self):  # type: ignore[override]
        self.setup_calls += 1
        return super()._setup_client_sync()

    def cleanup(self) -> None:  # type: ignore[override]
        self.cleanup_calls += 1
        return super().cleanup()


@pytest.mark.service
@pytest.mark.case_id("DAI-030")
def test_existing_connection_sync_and_cleanup(monkeypatch, text_endpoint, default_call_config):
    """
    GIVEN generate_with_existing_connection_sync is used for multiple calls
    WHEN cleanup_sync is invoked afterwards
    THEN the provider context should be entered once and cleaned exactly once
    """

    holder: dict[str, _CountingProvider] = {}

    def fake_provider(cls, model_endpoint, config, is_async):
        provider = _CountingProvider(model_endpoint=model_endpoint, config=config, is_async=is_async)
        holder["provider"] = provider
        return provider

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=default_call_config,
        is_async=False,
    )

    response_one = client.generate_with_existing_connection_sync(prompt="first")
    response_two = client.generate_with_existing_connection_sync(prompt="second")

    provider = holder["provider"]
    assert provider.setup_calls == 1
    assert response_one.status.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS
    assert response_two.status.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS

    client.cleanup_sync()

    assert provider.cleanup_calls == 1
    assert client._provider_client is None
