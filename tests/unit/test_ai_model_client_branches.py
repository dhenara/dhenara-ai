# Provenance: Added to improve AIModelClient branch coverage (2026-01-21)

from __future__ import annotations

import asyncio

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory

pytestmark = [pytest.mark.unit]


class _DummyResponse:
    def __init__(self, text: str):
        self._text = text

    def as_text(self) -> str:
        return self._text


class _ProviderCM:
    def __init__(self, response_text: str):
        self._response_text = response_text

        # AIModelClient probes for this and calls it start/stop.
        self.python_logs: list[str] = []

        def _capture(state: str):
            self.python_logs.append(state)

        self._capture_python_logs = _capture

    def format_and_generate_response_sync(self, *args, **kwargs):
        return _DummyResponse(self._response_text)

    async def format_and_generate_response_async(self, *args, **kwargs):
        return _DummyResponse(self._response_text)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_provider_cm(response_text: str):
    def _creator(cls, model_endpoint, config, is_async):
        return _ProviderCM(response_text)

    return _creator


@pytest.mark.case_id("DAI-065")
def test_dai_065_existing_connection_dispatch(monkeypatch, text_endpoint, default_call_config):
    """GIVEN sync and async AIModelClient instances
    WHEN generate_with_existing_connection is called
    THEN it dispatches to the correct sync/async implementation and returns a response.
    """

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(_make_provider_cm("ok")))

    sync_client = AIModelClient(model_endpoint=text_endpoint, config=default_call_config, is_async=False)

    async def _run_sync():
        resp = await sync_client.generate_with_existing_connection(prompt="hi")
        assert resp.as_text() == "ok"

    asyncio.run(_run_sync())
    sync_client.cleanup_sync()

    async_client = AIModelClient(model_endpoint=text_endpoint, config=default_call_config, is_async=True)

    async def _run():
        resp = await async_client.generate_with_existing_connection(prompt="hi")
        assert resp.as_text() == "ok"
        await async_client.cleanup_async()

    asyncio.run(_run())


@pytest.mark.case_id("DAI-066")
def test_dai_066_context_misuse_errors(monkeypatch, text_endpoint, default_call_config):
    """GIVEN sync and async AIModelClient instances
    WHEN generate/generate_async or context-managers are used incorrectly
    THEN it raises clear RuntimeError exceptions.
    """

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(_make_provider_cm("ok")))

    async_client = AIModelClient(model_endpoint=text_endpoint, config=default_call_config, is_async=True)
    with pytest.raises(RuntimeError, match=r"Use generate_async"):
        async_client.generate(prompt="hi")
    with pytest.raises(RuntimeError, match=r"Use 'async with'"):
        async_client.__enter__()

    sync_client = AIModelClient(model_endpoint=text_endpoint, config=default_call_config, is_async=False)

    async def _bad_async():
        with pytest.raises(RuntimeError, match=r"Use generate"):
            await sync_client.generate_async(prompt="hi")
        with pytest.raises(RuntimeError, match=r"Use 'with'"):
            await sync_client.__aenter__()

    asyncio.run(_bad_async())

    # Cleanup method misuse should also be rejected.
    with pytest.raises(RuntimeError, match=r"cleanup_sync called for an async client"):
        async_client.cleanup_sync()

    async def _bad_cleanup_async():
        with pytest.raises(RuntimeError, match=r"cleanup_async called for a sync client"):
            await sync_client.cleanup_async()

    asyncio.run(_bad_cleanup_async())
