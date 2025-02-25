import logging
from collections.abc import AsyncGenerator

from anthropic.types.message import Message
from dhenara.ai.providers.anthropic import AnthropicClientBase
from dhenara.ai.providers.common import StreamingManager
from dhenara.ai.types.external_api import (
    ExternalApiCallStatus,
)
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseContentItem,
    ChatResponseUsage,
    StreamingChatResponse,
    TokenStreamChunk,
)
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
class AnthropicChat(AnthropicClientBase):
    async def generate_response(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("inputs must be validated with validate_inputs() before api calls")

        parsed_response: ChatResponse | None = None
        api_call_status: ExternalApiCallStatus | None = None

        messages = []
        user = self.config.get_user()

        # Process system instructions
        system_prompt = self.process_instructions(instructions)

        # Add previous messages and current prompt
        if context:
            messages.extend(context)
        messages.append(prompt)

        # Prepare API call arguments
        chat_args = {
            "model": self.model_name_in_api_calls,
            "messages": messages,
            "stream": self.config.streaming,
        }

        if system_prompt:
            chat_args["system"] = system_prompt

        if user:
            chat_args["metadata"] = {"user_id": user}

        max_tokens = self.config.get_max_tokens(self.model_endpoint.ai_model)
        if max_tokens is not None:
            chat_args["max_tokens"] = max_tokens

        if self.config.options:
            chat_args.update(self.config.options)

        if self.config.test_mode:
            from dhenara.ai.providers.common.dummy import DummyAIModelResponseFns

            return await DummyAIModelResponseFns.get_dummy_ai_model_response(
                ai_model_ep=self.model_endpoint,
                streaming=self.config.streaming,
            )

        try:
            if self.config.streaming:
                stream = await self._client.messages.create(**chat_args)
                stream_generator = self._handle_streaming_response(stream=stream)
                return AIModelCallResponse(stream_generator=stream_generator)

            response = await self._client.messages.create(**chat_args)
            parsed_response = self.parse_response(response)
            api_call_status = self._create_success_status()

        except Exception as e:
            logger.exception(f"Error in generate_response: {e}")
            api_call_status = self._create_error_status(str(e))

        return AIModelCallResponse(
            status=api_call_status,
            chat_response=parsed_response,
        )

    async def _handle_streaming_response(
        self,
        stream: AsyncGenerator,
    ) -> AsyncGenerator[tuple[StreamingChatResponse | SSEErrorResponse, AIModelCallResponse | None]]:
        """Handle streaming response with progress tracking and final response"""
        stream_manager = StreamingManager(
            model_endpoint=self.model_endpoint,
        )

        try:
            async for chunk in stream:
                stream_response: StreamingChatResponse | None = None
                final_response: AIModelCallResponse | None = None
                stream_metadata = {}

                match chunk.type:
                    case "message_start":
                        # Initialize message metadata
                        stream_metadata = {
                            "message_id": chunk.message.id,
                            "model": chunk.message.model,
                            "type": chunk.type,
                        }
                        stream_manager.metadata["id"] = chunk.message.id

                        # Anthropic has a wieded way of reporint usage on streaming
                        _usage = chunk.message.usage
                        if _usage:
                            # Initialize usage in stream_manager
                            usage = ChatResponseUsage(
                                total_tokens=0,
                                prompt_tokens=_usage.input_tokens,
                                completion_tokens=_usage.output_tokens,
                            )
                            stream_manager.update_usage(usage)

                    case "content_block_delta":
                        # Content block update
                        delta_text = chunk.delta.text
                        if delta_text:
                            stream_manager.update(delta_text)

                            stream_response = StreamingChatResponse(
                                data=TokenStreamChunk(
                                    index=0,  # Anthropic doesn't provide choice index
                                    content=delta_text,
                                    done=False,
                                    metadata=stream_metadata,
                                ),
                            )

                    case "message_delta":
                        if hasattr(chunk, "usage") and chunk.usage:
                            stream_manager.usage.completion_tokens += chunk.usage.output_tokens

                        if hasattr(chunk.delta, "stop_reason"):
                            stream_manager.metadata["stop_reason"] = chunk.delta.stop_reason
                        if hasattr(chunk.delta, "stop_sequence"):
                            stream_manager.metadata["stop_sequence"] = chunk.delta.stop_sequence

                    case "message_delta":
                        _usage = chunk.delta.usage
                        if _usage:
                            stream_manager.usage.completion_tokens += _usage.output_tokens

                        stream_manager.metadata["stop_reason"] = chunk.delta.stop_reason
                        stream_manager.metadata["stop_sequence"] = chunk.delta.stop_sequence

                    case "message_stop":
                        stream_manager.usage.total_tokens = stream_manager.usage.prompt_tokens + stream_manager.usage.completion_tokens
                        stream_manager.complete()
                        final_response = stream_manager.get_final_response()

                        # Get final metadata for the last chunk
                        if final_response and final_response.chat_response:
                            stream_metadata = final_response.chat_response.get_visible_fields()

                        stream_response = StreamingChatResponse(
                            data=TokenStreamChunk(
                                index=0,
                                content="",  # Empty content for final chunk
                                done=True,
                                metadata=stream_metadata,
                            ),
                        )

                        yield stream_response, final_response
                        return  # Stop the generator

                # Yield stream response and final response if available
                if stream_response:
                    yield stream_response, final_response

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

    def _get_usage_from_provider_response(
        self,
        response: Message,
    ) -> ChatResponseUsage:
        return ChatResponseUsage(
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    def parse_response(self, response: Message) -> ChatResponse:
        usage, usage_charge = self.get_usage_and_charge(response)
        return ChatResponse(
            model=response.model,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            usage=usage,
            usage_charge=usage_charge,
            choices=[
                ChatResponseChoice(
                    index=index,
                    content=ChatResponseContentItem(
                        role=response.role,
                        text=content_item.text,
                    ),
                )
                for index, content_item in enumerate(response.content)
            ],
            metadata=AIModelCallResponseMetaData(
                streaming=False,
                duration_seconds=0,  # TODO
                provider_data={
                    "id": response.id,
                    "type": response.type,
                },
            ),
        )
