"""Service tests for AIModelClient timeout handling."""

import asyncio
import time

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types import AIModelCallResponse
from dhenara.ai.types.genai.dhenara.request import AIModelCallConfig


class _SlowSyncProvider:
    """Provider stub that sleeps longer than the configured timeout."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.call_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _format_and_generate_response_sync(self, *args, **kwargs):
        self.call_count += 1
        time.sleep(self.delay)
        return AIModelCallResponse()


class _SlowAsyncProvider:
    """Async provider stub that awaits longer than the configured timeout."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.call_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _format_and_generate_response_async(self, *args, **kwargs):
        self.call_count += 1
        await asyncio.sleep(self.delay)
        return AIModelCallResponse()


@pytest.mark.service
@pytest.mark.case_id("DAI-027")
def test_sync_timeout_raises_and_cancels(monkeypatch, text_endpoint):
    """
    GIVEN a synchronous provider that exceeds the configured timeout
    WHEN generate() is invoked
    THEN a TimeoutError should be raised after the first attempt
    AND the slow call should not complete execution
    """

    slow_provider = _SlowSyncProvider(delay=0.05)

    def fake_provider(cls, model_endpoint, config, is_async):
        assert not is_async
        return slow_provider

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=AIModelCallConfig(timeout=0.01, retries=1),
        is_async=False,
    )

    with pytest.raises(TimeoutError):
        client.generate(prompt="slow")

    assert slow_provider.call_count == 1


@pytest.mark.service
@pytest.mark.case_id("DAI-028")
@pytest.mark.asyncio
async def test_async_timeout_raises(monkeypatch, text_endpoint):
    """
    GIVEN an async provider that exceeds the configured timeout
    WHEN generate_async() is invoked
    THEN a TimeoutError should be raised after the first attempt
    AND the slow coroutine should be cancelled
    """

    slow_provider = _SlowAsyncProvider(delay=0.05)

    def fake_provider(cls, model_endpoint, config, is_async):
        assert is_async
        return slow_provider

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=AIModelCallConfig(timeout=0.01, retries=1),
        is_async=True,
    )

    with pytest.raises(TimeoutError):
        await client.generate_async(prompt="slow")

    assert slow_provider.call_count == 1
