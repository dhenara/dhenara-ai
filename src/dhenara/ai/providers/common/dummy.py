import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Union

from dhenara.ai.config import settings
from dhenara.ai.providers.base import StreamingManager
from dhenara.ai.types.external_api import (
    ExternalApiCallStatus,
    ExternalApiCallStatusEnum,
)
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    AIModelFunctionalTypeEnum,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseTextContentItem,
    ChatResponseTextContentItemDelta,
    ChatResponseUsage,
    ImageResponseUsage,
    StreamingChatResponse,
    UsageCharge,
)
from dhenara.ai.types.genai.ai_model import AIModelEndpoint
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
class DummyAIModelResponseFns:
    @staticmethod
    async def get_dummy_ai_model_response(
        ai_model_ep: AIModelEndpoint,
        streaming: bool = False,
    ) -> AIModelCallResponse:
        if streaming:
            stream_generator = DummyAIModelResponseFns._handle_streaming_response(
                stream=DummyAIModelResponseFns.gen_dummy_stream(ai_model_ep),
                model_endpoint=ai_model_ep,
            )

            return AIModelCallResponse(
                async_stream_generator=stream_generator,
            )

        text = f"This is a test mode output. {DummyAIModelResponseFns._model_specific_message(ai_model_ep)}"
        api_call_status = ExternalApiCallStatus(
            status=ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS,
            api_provider="dummu_ai_provider",
            model="dummy_model",
            message="Output generated",
            code="success",
            http_status_code=200,
        )
        usage, usage_charge = DummyAIModelResponseFns.get_dummy_usage_and_charge(ai_model_ep=ai_model_ep)
        parsed_response = ChatResponse(
            model="dummy_model",
            provider=ai_model_ep.ai_model.provider,
            api_provider=ai_model_ep.api.provider,
            usage=usage,
            usage_charge=usage_charge,
            choices=[
                ChatResponseChoice(
                    index=0,
                    contents=[
                        ChatResponseTextContentItem(
                            role="assistant",
                            text=text,
                        )
                    ],
                )
            ],
            metadata=AIModelCallResponseMetaData(
                streaming=False,
                duration_seconds=0,  # TODO
                provider_metadata={
                    "id": "test_msg_id",
                    "object": "test_object",
                    "created": "",
                    "system_fingerprint": "",
                },
            ),
        )

        return AIModelCallResponse(
            status=api_call_status,
            chat_response=parsed_response,
        )

    # For Usage and cost
    @staticmethod
    def get_dummy_usage_and_charge(
        ai_model_ep: AIModelEndpoint,
    ) -> tuple[Union[ChatResponseUsage, ImageResponseUsage, None], Union[UsageCharge | None]]:
        """Parse the OpenAI response into our standard format"""
        usage = None
        usage_charge = None

        if settings.ENABLE_USAGE_TRACKING or settings.ENABLE_COST_TRACKING:
            if ai_model_ep.ai_model.functional_type == AIModelFunctionalTypeEnum.TEXT_GENERATION:
                usage = ChatResponseUsage(total_tokens=1222, prompt_tokens=1000, completion_tokens=222)
            else:
                usage = ImageResponseUsage(number_of_images=1, model=ai_model_ep.ai_model.model_name, options={})

            if settings.ENABLE_COST_TRACKING:
                usage_charge = UsageCharge(cost=0.11, charge=0.22)

        return (usage, usage_charge)

    # -------------------------------------------------------------------------
    @staticmethod
    def _model_specific_message(
        ai_model_ep: AIModelEndpoint,
    ):
        return f"This test message is from Model {ai_model_ep.ai_model.model_name}:{ai_model_ep.ai_model.provider} with API Provider {ai_model_ep.api.provider}"

    # -------------------------------------------------------------------------
    @staticmethod
    async def _handle_streaming_response(
        stream: AsyncGenerator,
        model_endpoint: AIModelEndpoint,
    ) -> AsyncGenerator[tuple[StreamingChatResponse | SSEErrorResponse, AIModelCallResponse | None]]:
        """Handle streaming response with progress tracking and final response"""
        stream_manager = StreamingManager(
            model_endpoint=model_endpoint,
        )
        _second_last_chunk = None
        _last_chunk = None

        try:
            async for chunk in stream:
                stream_response: StreamingChatResponse | None = None
                final_response: AIModelCallResponse | None = None

                # Process usage
                if chunk.usage:
                    usage = ChatResponseUsage(
                        total_tokens=chunk.usage["total_tokens"],
                        prompt_tokens=chunk.usage["prompt_tokens"],
                        completion_tokens=chunk.usage["completion_tokens"],
                    )
                    stream_manager.update_usage(usage)

                if _second_last_chunk:
                    _last_chunk = chunk

                # Process content
                if chunk.choices:
                    choice_deltas = [
                        ChatResponseChoiceDelta(
                            index=choice.index,
                            finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None,
                            content_deltas=[
                                ChatResponseTextContentItemDelta(
                                    index=0,
                                    role="assistant",
                                    text_delta=choice.delta.content or "xyz",
                                )
                            ],
                            metadata={},
                        )
                        for choice in chunk.choices
                    ]

                    response_chunk = stream_manager.update(choice_deltas=choice_deltas)
                    stream_response = StreamingChatResponse(
                        id=None,
                        data=response_chunk,
                    )

                    yield stream_response, final_response

            # API has stopped streaming, get final response
            logger.debug("API has stopped streaming, processsing final response")
            final_response = stream_manager.get_final_response()
            yield None, final_response
            return  # Stop the generator
        except Exception as e:
            logger.exception(f"Error during streaming: {e}")
            error_response = SSEErrorResponse(
                data=SSEErrorData(
                    error_code=SSEErrorCode.external_api_error,
                    message="External API Server Error",
                    details={"error_type": type(e).__name__},
                )
            )
            yield error_response, None

    # -------------------------------------------------------------------------
    @staticmethod
    async def gen_dummy_stream(
        ai_model_ep: AIModelEndpoint,
    ):
        """Test implementation of AI model streaming response"""
        small_text = f"One\n Two\n Three\n Four\n Five\n Six Seven Eight Nine Ten"  # noqa: F541, F841
        large_text = """Here is a detailed analysis of the topic:

    1. First Important Point
       - Key insight A
       - Supporting detail B
       - Related concept C

    2. Second Important Point
       - Technical aspect X
       - Implementation Y
       - Best practice Z

    3. Final Considerations
       - Future implications
       - Recommendations
       - Next steps

    Let me know if you need any clarification!"""  # noqa: F841, RUF100

        class DummyChunk:
            def __init__(self, content, index=0, finish=False):
                self.id = "dummy-stream-id"
                self.object = "chat.completion.chunk"
                self.created = 1738776944
                self.system_fingerprint = "fp_dummy"
                self.choices = [type("Choice", (), {"index": index, "delta": type("Delta", (), {"content": content}), "finish_reason": "stop" if finish else None})()]
                self.usage = None if not finish else {"total_tokens": 1000, "prompt_tokens": 333, "completion_tokens": 777}

        text = f"{DummyAIModelResponseFns._model_specific_message(ai_model_ep)}\n{large_text}"
        words = text.split(" ")
        for i, word in enumerate(words):
            await asyncio.sleep(0.04)  # Simulate network delay
            is_last = i == len(words) - 1
            yield DummyChunk(word + " ", index=0, finish=is_last)
