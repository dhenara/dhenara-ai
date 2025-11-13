"""Component tests for DummyAIModelResponseFns test-mode provider."""

import pytest

from dhenara.ai.providers.base.streaming_manager import StreamingManager
from dhenara.ai.providers.common.dummy import DummyAIModelResponseFns
from dhenara.ai.types import ExternalApiCallStatusEnum, StreamingChatResponse


@pytest.mark.component
@pytest.mark.case_id("DAI-019")
def test_sync_non_streaming_response_fields(text_endpoint):
    """
    GIVEN the dummy provider in non-streaming mode
    WHEN get_dummy_ai_model_response_sync is invoked
    THEN it should return a successful ChatResponse with usage metadata populated
    """

    streaming_manager = StreamingManager(model_endpoint=text_endpoint, structured_output_config=None)
    dummy = DummyAIModelResponseFns(streaming_manager=streaming_manager)

    response = dummy.get_dummy_ai_model_response_sync(text_endpoint, streaming=False)

    if response.status.status != ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS:
        pytest.fail("Dummy provider should report success for non-streaming sync calls")

    if response.chat_response is None:
        pytest.fail("Dummy provider must return a chat response instance")

    text = response.chat_response.text()
    if text is None or "test mode output" not in text:
        pytest.fail("Dummy provider response text should indicate test mode output")

    if response.chat_response.usage is None:
        pytest.fail("Dummy provider response should include usage metadata")


@pytest.mark.component
@pytest.mark.case_id("DAI-020")
def test_streaming_sync_chunks_and_final(text_endpoint):
    """
    GIVEN the dummy provider in streaming mode (sync)
    WHEN consuming the sync_stream_generator
    THEN streaming chunks should be emitted followed by a final aggregated response
    """

    streaming_manager = StreamingManager(model_endpoint=text_endpoint, structured_output_config=None)
    dummy = DummyAIModelResponseFns(streaming_manager=streaming_manager)

    response = dummy.get_dummy_ai_model_response_sync(text_endpoint, streaming=True)
    stream = response.sync_stream_generator

    chunks = []
    final_response = None
    for chunk, full in stream:
        if chunk is not None:
            chunks.append(chunk)
        if full is not None:
            final_response = full
            break

    if not any(isinstance(chunk, StreamingChatResponse) for chunk in chunks):
        pytest.fail("Streaming sync generator should emit StreamingChatResponse chunks")

    if final_response is None:
        pytest.fail("Streaming sync generator should yield a final aggregated response")

    if final_response.chat_response is None:
        pytest.fail("Final streaming sync response must include a chat response")


@pytest.mark.component
@pytest.mark.case_id("DAI-021")
@pytest.mark.asyncio
async def test_streaming_async_chunks_and_final(text_endpoint):
    """
    GIVEN the dummy provider in streaming mode (async)
    WHEN consuming the async_stream_generator
    THEN streaming chunks should arrive and a final response should conclude the stream
    """

    streaming_manager = StreamingManager(model_endpoint=text_endpoint, structured_output_config=None)
    dummy = DummyAIModelResponseFns(streaming_manager=streaming_manager)

    response = await dummy.get_dummy_ai_model_response_async(text_endpoint, streaming=True)
    stream = response.async_stream_generator

    chunks = []
    final_response = None
    async for chunk, full in stream:
        if chunk is not None:
            chunks.append(chunk)
        if full is not None:
            final_response = full
            break

    if not any(isinstance(chunk, StreamingChatResponse) for chunk in chunks):
        pytest.fail("Streaming async generator should emit StreamingChatResponse chunks")

    if final_response is None:
        pytest.fail("Streaming async generator should yield a final aggregated response")

    if final_response.chat_response is None:
        pytest.fail("Final streaming async response must include a chat response")
