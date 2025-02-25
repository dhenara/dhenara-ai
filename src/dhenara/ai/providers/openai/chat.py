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
    ChatResponseUsage,
    StreamingChatResponse,
    TokenStreamChunk,
)
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse
from openai.types.chat import ChatCompletion

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

        max_tokens = self.config.get_max_tokens(self.model_endpoint.ai_model)
        if max_tokens is not None:
            if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
                # NOTE: With resasoning models, max_tokens Deprecated in favour of max_completion_tokens
                chat_args["max_completion_tokens"] = max_tokens

            else:
                chat_args["max_tokens"] = max_tokens

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
            model_name=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
        )

        try:
            async for chunk in stream:
                # Second last chunk will have usage
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

                # Process content
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta.content or ""
                    stream_metadata = {}

                    # Update streaming progress
                    stream_manager.update(delta)

                    # Prepare streaming response
                    if choice.finish_reason:
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
                        final_response = stream_manager.get_final_response()
                        stream_metadata = final_response.chat_response.get_visible_fields()

                # Create StreamingChatResponse
                stream_response = StreamingChatResponse(
                    id=None,  # str(uuid.uuid4()),
                    data=TokenStreamChunk(
                        index=choice.index,
                        content=delta,
                        done=bool(choice.finish_reason),
                        metadata=stream_metadata,
                    ),
                )

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
                    content=ChatResponseContentItem(
                        role=choice.message.role,
                        text=choice.message.content,
                        function_call=choice.message.function_call if hasattr(choice.message, "function_call") else None,
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
