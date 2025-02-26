import logging
from collections.abc import AsyncGenerator
from typing import Any

from dhenara.ai.providers.common import StreamingManager
from dhenara.ai.providers.openai import OpenAIClientBase
from dhenara.ai.types.external_api import (
    AIModelAPIProviderEnum,
    ExternalApiCallStatus,
)
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseContentItem,
    ChatResponseGenericContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCallContentItem,
    ChatResponseUsage,
    StreamingChatResponse,
    TokenStreamChunk,
)
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse
from openai.types.chat import ChatCompletion, ChatCompletionMessage

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
class OpenAIChat(OpenAIClientBase):
    async def generate_response(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("inputs must be validated with `self.validate_inputs()` before api calls")

        parsed_response: ChatResponse | None = None
        api_call_status: ExternalApiCallStatus | None = None

        messages = []
        user = self.config.get_user()

        # Process instructions
        instructions_prompt = self.process_instructions(instructions)
        if instructions_prompt:
            messages.append(instructions_prompt)

        # Add previous messages and current prompt
        if context:
            messages.extend(context)
        messages.append(prompt)

        # Prepare API call arguments
        chat_args: dict[str, Any] = {
            "model": self.model_name_in_api_calls,
            "messages": messages,
            "stream": self.config.streaming,
        }

        if user:
            if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
                chat_args["user"] = user

        max_output_tokens, max_reasoning_tokens = self.config.get_max_output_tokens(self.model_endpoint.ai_model)

        if max_output_tokens is not None:
            if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
                # NOTE: With resasoning models, max_output_tokens Deprecated in favour of max_completion_tokens
                chat_args["max_completion_tokens"] = max_output_tokens

            else:
                chat_args["max_tokens"] = max_output_tokens

        if self.config.streaming:
            if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
                chat_args["stream_options"] = {"include_usage": True}

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
                if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
                    stream = await self._client.chat.completions.create(**chat_args)
                else:
                    stream = await self._client.complete(**chat_args)

                stream_generator = self._handle_streaming_response(
                    stream=stream,
                )
                return AIModelCallResponse(
                    stream_generator=stream_generator,
                )

            if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
                response = await self._client.chat.completions.create(**chat_args)
            else:
                response = await self._client.complete(**chat_args)

            parsed_response = self.parse_response(
                response=response,
            )
            api_call_status = self._create_success_status()

        except Exception as e:
            logger.exception(f"Error in get_model_response: {e}")
            api_call_status = self._create_error_status(str(e))

        return AIModelCallResponse(
            status=api_call_status,
            chat_response=parsed_response,
        )

    # -------------------------------------------------------------------------
    async def _handle_streaming_response(
        self,
        stream: Any,
    ) -> AsyncGenerator[tuple[StreamingChatResponse | SSEErrorResponse, AIModelCallResponse | None]]:
        """Handle streaming response with progress tracking and final response"""
        stream_manager = StreamingManager(
            model_endpoint=self.model_endpoint,
        )
        _second_last_chunk = None
        _last_chunk = None

        try:
            async for chunk in stream:
                stream_response: StreamingChatResponse | None = None
                final_response: AIModelCallResponse | None = None

                # Process usage
                if hasattr(chunk, "usage") and chunk.usage:  # Microsoft is slow in adopting openai changes 😶
                    usage = ChatResponseUsage(
                        total_tokens=chunk.usage.total_tokens,
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                    )
                    stream_manager.update_usage(usage)

                if _second_last_chunk:
                    _last_chunk = chunk

                # Process content
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta.content or ""

                    # Update streaming progress
                    stream_manager.update(delta)

                    is_final_chunk = bool(choice.finish_reason)
                    # Prepare streaming response
                    if is_final_chunk:
                        _second_last_chunk = chunk

                        stream_manager.complete(
                            metadata={
                                "id": chunk.id,
                                "created": str(chunk.created),  # Microsoft sdk returns datetim obj
                                "object": chunk.object if hasattr(chunk, "object") else None,
                                "system_fingerprint": chunk.system_fingerprint if hasattr(chunk, "system_fingerprint") else None,
                                # TODO_FUTURE: Support  multiple chat completion choices, and move finish reason to choice
                                "finish_reason": choice.finish_reason if hasattr(choice, "finish_reason") else None,
                            },
                        )

                    # Create StreamingChatResponse
                    stream_response = StreamingChatResponse(
                        id=None,
                        data=TokenStreamChunk(
                            index=choice.index,
                            content=delta,
                            done=is_final_chunk,
                            metadata={},
                        ),
                    )

                    yield stream_response, final_response

                if _last_chunk:
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
    def _get_usage_from_provider_response(
        self,
        response: ChatCompletion,
    ) -> ChatResponseUsage:
        return ChatResponseUsage(
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    # -------------------------------------------------------------------------
    def parse_response(
        self,
        response: ChatCompletion,
    ) -> ChatResponse:
        """Parse the OpenAI response into our standard format"""
        usage, usage_charge = self.get_usage_and_charge(response)

        return ChatResponse(
            model=response.model,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            usage=usage,
            usage_charge=usage_charge,
            choices=[
                ChatResponseChoice(
                    index=choice.index,
                    finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None,
                    contents=self.process_choice(
                        role=choice.message.role,
                        choice=choice,
                    ),
                )
                for choice in response.choices
            ],
            metadata=AIModelCallResponseMetaData(
                streaming=False,
                duration_seconds=0,  # TODO
                provider_data={
                    "id": response.id,
                    "created": str(response.created),  # Microsoft sdk returns datetim obj
                    "object": response.object if hasattr(response, "object") else None,
                    "system_fingerprint": response.system_fingerprint if hasattr(response, "system_fingerprint") else None,
                },
            ),
        )

    def process_choice(self, role, choice) -> list[ChatResponseContentItem]:
        return [
            # self.process_content_item(
            #    role=role,
            #    content_item=content_item,
            # )
            # for content_item in choice.message
            #
            #
            # Only one content item in choice.message, might change with reasoning responses ?
            self.process_content_item(
                role=role,
                content_item=choice.message,
            )
        ]

    def process_content_item(self, role, content_item: ChatCompletionMessage) -> ChatResponseContentItem:
        if isinstance(content_item, ChatCompletionMessage):
            if content_item.content:
                return ChatResponseTextContentItem(
                    role=role,
                    text=content_item.content,
                )
            elif content_item.tool_calls:
                return ChatResponseToolCallContentItem(
                    role=role,
                    metadata={"tool_calls": [tool_call.model_dump() for tool_call in content_item.tool_calls]},
                )
            else:
                return ChatResponseGenericContentItem(
                    role=role,
                    metadata={"part": content_item.model_dump()},
                )
        else:
            logger.fatal(f"process_content_item: Unknown content item type {type(content_item)}")
            return ChatResponseGenericContentItem(
                role=role,
                metadata={
                    "unknonwn": f"Unknown content item of type {type(content_item)}",
                },
            )
