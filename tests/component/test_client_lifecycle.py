"""Component tests covering AIModelClient context manager behaviour."""

import asyncio

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types import AIModelCallResponse


class _FakeSyncProvider:
    """Minimal sync provider stub for testing client lifecycle."""

    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_count += 1
        return False

    def _format_and_generate_response_sync(self, *args, **kwargs) -> AIModelCallResponse:
        return AIModelCallResponse()


class _FakeAsyncProvider:
    """Minimal async provider stub for testing async client lifecycle."""

    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_count += 1
        return False

    async def _format_and_generate_response_async(self, *args, **kwargs) -> AIModelCallResponse:
        return AIModelCallResponse()


@pytest.mark.component
@pytest.mark.case_id("DAI-013")
def test_sync_context_and_errors_for_async(monkeypatch, text_endpoint, default_call_config):
    """
    GIVEN an AIModelClient configured for synchronous usage
    WHEN used with the context manager
    THEN it should enter and exit cleanly, and reject async context usage
    """

    provider = _FakeSyncProvider()

    def fake_create_provider(cls, model_endpoint, config, is_async):
        assert is_async is False
        return provider

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_create_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=default_call_config,
        is_async=False,
    )

    with client as active_client:
        assert active_client is client
        assert provider.enter_count == 1

    assert provider.exit_count == 1
    assert client._provider_client is None

    async def attempt_async_context():
        async with client:
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(attempt_async_context())


@pytest.mark.component
@pytest.mark.case_id("DAI-014")
@pytest.mark.asyncio
async def test_async_context_and_errors_for_sync(monkeypatch, text_endpoint, default_call_config):
    """
    GIVEN an AIModelClient configured for async usage
    WHEN used with an async context manager
    THEN it should enter and exit cleanly, and reject sync context usage
    """

    provider = _FakeAsyncProvider()

    def fake_create_provider(cls, model_endpoint, config, is_async):
        assert is_async is True
        return provider

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_create_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=default_call_config,
        is_async=True,
    )

    async with client as active_client:
        assert active_client is client
        assert provider.enter_count == 1

    assert provider.exit_count == 1
    assert client._provider_client is None

    with pytest.raises(RuntimeError):
        with client:
            pass
