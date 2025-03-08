import asyncio
import logging
from collections.abc import AsyncGenerator, Generator

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
    def __init__(self, streaming_manager: StreamingManager):
        self.streaming_manager = streaming_manager

    def _create_base_response(self, ai_model_ep: AIModelEndpoint, text: str) -> AIModelCallResponse:
        """Create a base response with common elements"""
        api_call_status = ExternalApiCallStatus(
            status=ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS,
            api_provider="dummy_ai_provider",
            model="dummy_model",
            message="Output generated",
            code="success",
            http_status_code=200,
        )

        usage, usage_charge = self.get_dummy_usage_and_charge(ai_model_ep)

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
                duration_seconds=0,
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

    def get_dummy_ai_model_response_sync(
        self,
        ai_model_ep: AIModelEndpoint,
        streaming: bool = False,
    ) -> AIModelCallResponse:
        if streaming:
            _stream_generator = self.handle_streaming_response_sync(
                self.sync_stream_generator(ai_model_ep),
                model_endpoint=ai_model_ep,
            )
            return AIModelCallResponse(sync_stream_generator=_stream_generator)
        return self._create_base_response(
            ai_model_ep,
            f"This is a test mode output. {self._model_specific_message(ai_model_ep)}",
        )

    async def get_dummy_ai_model_response_async(
        self,
        ai_model_ep: AIModelEndpoint,
        streaming: bool = False,
    ) -> AIModelCallResponse:
        if streaming:
            _stream_generator = self.handle_streaming_response_async(
                self.async_stream_generator(ai_model_ep),
                model_endpoint=ai_model_ep,
            )
            return AIModelCallResponse(async_stream_generator=_stream_generator)
        return self._create_base_response(
            ai_model_ep,
            f"This is a test mode output. {self._model_specific_message(ai_model_ep)}",
        )

    def get_dummy_usage_and_charge(
        self,
        ai_model_ep: AIModelEndpoint,
    ) -> tuple[ChatResponseUsage | ImageResponseUsage | None, UsageCharge | None]:
        """Get dummy usage and charge data"""
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

    def _model_specific_message(
        self,
        ai_model_ep: AIModelEndpoint,
    ):
        return (
            f"This test message is from Model {ai_model_ep.ai_model.model_name}:"
            f"{ai_model_ep.ai_model.provider} with API Provider {ai_model_ep.api.provider}"
        )

    def _create_dummy_chunk(self, content: str, index: int = 0, finish: bool = False):
        """Create a dummy chunk for both sync and async streaming"""
        return type(
            "DummyChunk",
            (),
            {
                "id": "dummy-stream-id",
                "object": "chat.completion.chunk",
                "created": 1738776944,
                "system_fingerprint": "fp_dummy",
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "index": index,
                            "delta": type("Delta", (), {"content": content}),
                            "finish_reason": "stop" if finish else None,
                        },
                    )()
                ],
                "usage": None if not finish else {"total_tokens": 1000, "prompt_tokens": 333, "completion_tokens": 777},
            },
        )

    def _get_stream_text(self, ai_model_ep: AIModelEndpoint) -> str:
        """Get the text to be streamed"""
        base_text = self._model_specific_message(ai_model_ep)

        small_text = f"One\n Two\n Three\n Four\n Five\n Six Seven Eight Nine Ten"  # noqa: F541, F841
        large_text = """

Here is a detailed analysis of the topic:

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

Let me know if you need any clarification!"""

        markdown_text = """"
The issue with newlines being stripped in your code appears to be in how you're handling the incoming text deltas.
Looking at your logs, the newline characters (\n) are actually present in the `text_delta` values.

Here are a few potential solutions:

1. First, ensure your `onTextToken` callback preserves the newlines when displaying or processing the text. For example:

```javascript
onTextToken: (choice_index, content_index, text) => {
  // Preserve newlines by using pre-formatted elements if displaying in HTML
  const formattedText = text.replace(/\n/g, '<br>');
  // Or if appending to a textarea/console
  console.log(text); // Don't modify the text at all
},
```

2. If you're concatenating the tokens, make sure you're not accidentally stripping newlines:

```javascript
let fullText = '';
const processor = new StreamProcessor({
  onTextToken: (choice_index, content_index, text) => {
    fullText += text; // Don't modify the text, append as-is
    // If displaying in HTML
    element.innerHTML = fullText.replace(/\n/g, '<br>');
  }
});
```

3. You might also want to modify your `parseSSEEvent` method to explicitly preserve newlines:

```javascript
parseSSEEvent(eventString) {
  const lines = eventString.split(/\r?\n/);
  // ... rest of your parsing code ...

  if (data.length) {
    try {
      const dataString = data.join('\n'); // Preserve newlines when joining
      event.data = JSON.parse(dataString);
    } catch (e) {
      console.error('Error parsing JSON:', e);
      return null;
    }
  }

  return event;
}
```


Looking at your log output, there are tokens containing newlines,
so the issue is likely in how you're handling these tokens after receiving them rather than in the streaming process.

        """  # noqa: F841
        return f"{base_text}\n\n\n{large_text}"

    def _create_streaming_response(
        self, chunk_deltas: list[ChatResponseChoiceDelta]
    ) -> tuple[StreamingChatResponse | None, AIModelCallResponse | None]:
        """Process streaming chunks and create appropriate response"""
        stream_response = None

        if chunk_deltas:
            response_chunk = self.streaming_manager.update(choice_deltas=chunk_deltas)
            stream_response = StreamingChatResponse(
                id=None,
                data=response_chunk,
            )

        return stream_response

    def _process_chunk(self, chunk) -> StreamingChatResponse | None:
        """Process a single chunk for both sync and async streaming"""
        if chunk.usage:
            usage = ChatResponseUsage(
                total_tokens=chunk.usage["total_tokens"],
                prompt_tokens=chunk.usage["prompt_tokens"],
                completion_tokens=chunk.usage["completion_tokens"],
            )
            self.streaming_manager.update_usage(usage)

        if chunk.choices:
            choice_deltas = [
                ChatResponseChoiceDelta(
                    index=choice.index,
                    finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None,
                    content_deltas=[
                        ChatResponseTextContentItemDelta(
                            index=0,
                            role="assistant",
                            text_delta=choice.delta.content or "",
                        )
                    ],
                    metadata={},
                )
                for choice in chunk.choices
            ]
            return self._create_streaming_response(choice_deltas)
        return None, None

    def sync_stream_generator(self, ai_model_ep: AIModelEndpoint):
        """Synchronous streaming generator"""
        import time

        text = self._get_stream_text(ai_model_ep)
        words = text.split(" ")

        for i, word in enumerate(words):
            time.sleep(0.04)  # Simulate network delay
            is_last = i == len(words) - 1
            yield self._create_dummy_chunk(word + " ", index=0, finish=is_last)

    async def async_stream_generator(self, ai_model_ep: AIModelEndpoint):
        """Asynchronous streaming generator"""
        text = self._get_stream_text(ai_model_ep)
        words = text.split(" ")

        for i, word in enumerate(words):
            await asyncio.sleep(0.04)  # Simulate network delay
            is_last = i == len(words) - 1
            yield self._create_dummy_chunk(word + " ", index=0, finish=is_last)

    def handle_streaming_response_sync(
        self,
        stream_generator,
        model_endpoint: AIModelEndpoint,
    ) -> Generator[tuple[StreamingChatResponse | SSEErrorResponse, AIModelCallResponse | None], None, None]:
        """Handle synchronous streaming response"""

        try:
            for chunk in stream_generator:
                stream_response = self._process_chunk(chunk)
                if stream_response:
                    yield stream_response, None

            # API has stopped streaming, send done-chunk and get final response
            done_chunk = self.streaming_manager.get_streaming_done_chunk()
            yield done_chunk, None

            # Final response after stream ends
            final_response = self.streaming_manager.get_final_response()
            yield None, final_response
            return

        except Exception as e:
            logger.exception(f"Error during streaming: {e}")
            yield (
                SSEErrorResponse(
                    data=SSEErrorData(
                        error_code=SSEErrorCode.external_api_error,
                        message="External API Server Error",
                        details={"error_type": type(e).__name__},
                    )
                ),
                None,
            )

    async def handle_streaming_response_async(
        self,
        stream_generator,
        model_endpoint: AIModelEndpoint,
    ) -> AsyncGenerator[tuple[StreamingChatResponse | SSEErrorResponse, AIModelCallResponse | None], None]:
        """Handle asynchronous streaming response"""

        try:
            async for chunk in stream_generator:
                stream_response = self._process_chunk(chunk)
                if stream_response:
                    yield stream_response, None

            # API has stopped streaming, send done-chunk and get final response
            done_chunk = self.streaming_manager.get_streaming_done_chunk()
            yield done_chunk, None

            # Final response after stream ends
            final_response = self.streaming_manager.get_final_response()
            yield None, final_response
            return

        except Exception as e:
            logger.exception(f"Error during streaming: {e}")
            yield (
                SSEErrorResponse(
                    data=SSEErrorData(
                        error_code=SSEErrorCode.external_api_error,
                        message="External API Server Error",
                        details={"error_type": type(e).__name__},
                    )
                ),
                None,
            )
