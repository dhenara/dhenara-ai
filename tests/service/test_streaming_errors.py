"""Service-level streaming error handling tests."""

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types import AIModelCallResponse
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse

from ..helpers import FakeProvider


class _ErrorStreamingProvider(FakeProvider):
    """Fake provider that emits a streaming error immediately."""

    def do_streaming_api_call_sync(self, api_call_params):  # type: ignore[override]
        def _gen():
            yield {"type": "error"}

        return _gen()

    def parse_stream_chunk(self, chunk):  # type: ignore[override]
        return [
            SSEErrorResponse(
                data=SSEErrorData(
                    error_code=SSEErrorCode.external_api_error,
                    message="External API Server Error",
                    details={"chunk": chunk},
                )
            )
        ]

    def do_api_call_sync(self, api_call_params):  # type: ignore[override]
        return AIModelCallResponse()


@pytest.mark.service
@pytest.mark.case_id("DAI-031")
def test_sse_error_stops_stream_and_logs(monkeypatch, text_endpoint, default_call_config):
    """
    GIVEN a streaming provider that produces an SSE error chunk
    WHEN generate() is invoked with streaming enabled
    THEN the first streamed tuple should contain the SSEErrorResponse and terminate the stream
    """

    config = default_call_config.model_copy(update={"streaming": True})

    def fake_provider(cls, model_endpoint, config, is_async):
        return _ErrorStreamingProvider(model_endpoint=model_endpoint, config=config, is_async=is_async)

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=config,
        is_async=False,
    )

    response = client.generate(prompt="stream")
    stream = response.sync_stream_generator
    assert stream is not None

    first_chunk, final_response = next(stream)
    assert isinstance(first_chunk, SSEErrorResponse)
    assert final_response is None

    with pytest.raises(StopIteration):
        next(stream)
