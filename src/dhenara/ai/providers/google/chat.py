import logging

# Copyright 2024-2025 Dhenara Inc. All rights reserved.
from collections.abc import AsyncGenerator
from typing import Any

from dhenara.ai.libs.fns import generic_obj_to_dict
from dhenara.ai.providers.common import StreamingManager
from dhenara.ai.providers.google import GoogleAIClientBase
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
from google.genai.types import GenerateContentConfig, GenerateContentResponse, SafetySetting

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


models_not_supporting_system_instructions = ["gemini-1.0-pro"]


# -----------------------------------------------------------------------------
class GoogleAIChat(GoogleAIClientBase):
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

        generate_config_args = self.get_default_generate_config_args()
        generate_config = GenerateContentConfig(**generate_config_args)

        # Process instructions
        instructions_str = self.process_instructions(instructions)
        if isinstance(instructions_str, dict):
            if context:
                context.insert(0, instructions_str)
            else:
                context = [instructions_str]
        elif instructions_str and not any(self.model_endpoint.ai_model.model_name.startswith(model) for model in ["gemini-1.0-pro"]):
            generate_config.system_instruction = instructions_str

        if self.config.test_mode:
            from dhenara.ai.providers.common.dummy import DummyAIModelResponseFns

            return await DummyAIModelResponseFns.get_dummy_ai_model_response(
                ai_model_ep=self.model_endpoint,
                streaming=self.config.streaming,
            )

        try:
            chat = self._client.chats.create(
                model=self.model_name_in_api_calls,
                config=generate_config,
                history=context or [],
            )

            if self.config.streaming:
                stream = await chat.send_message_stream(
                    message=prompt,
                )
                stream_generator = self._handle_streaming_response(
                    stream=stream,
                )
                return AIModelCallResponse(
                    stream_generator=stream_generator,
                )

            response = await chat.send_message(
                message=prompt,
            )
            parsed_response = self.parse_response(response)
            api_call_status = self._create_success_status()

        except Exception as e:
            logger.exception(f"Error in generate_response: {e}")
            api_call_status = self._create_error_status(str(e))

        return AIModelCallResponse(
            status=api_call_status,
            chat_response=parsed_response,
        )

    def get_default_generate_config_args(self) -> dict:
        max_tokens = self.config.get_max_tokens(self.model_endpoint.ai_model)
        safety_settings = [
            SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            )
        ]

        config_params = {
            "candidate_count": 1,
            "safety_settings": safety_settings,
        }

        if max_tokens:
            config_params["max_output_tokens"] = max_tokens

        return config_params

    async def _handle_streaming_response(
        self,
        stream: Any,
    ) -> AsyncGenerator[tuple[StreamingChatResponse | SSEErrorResponse, AIModelCallResponse | None]]:
        stream_manager = StreamingManager(
            model_endpoint=self.model_endpoint,
        )

        try:
            async for chunk in stream:
                stream_response: StreamingChatResponse | None = None
                final_response: AIModelCallResponse | None = None

                # Process content from candidates
                if chunk.candidates:
                    for candidate in chunk.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.text:
                                    stream_manager.update(part.text)

                                    # Check if this is the final chunk
                                    is_done = bool(candidate.finish_reason)

                                    stream_response = StreamingChatResponse(
                                        id=None,
                                        data=TokenStreamChunk(
                                            index=candidate.index or 0,
                                            content=part.text,
                                            done=is_done,
                                            metadata={},
                                        ),
                                    )

                                    if is_done:
                                        stream_manager.complete(
                                            metadata={
                                                "prompt_feedback": chunk.prompt_feedback,
                                                "finish_reason": candidate.finish_reason,
                                                "safety_ratings": candidate.safety_ratings,
                                            },
                                        )
                                        usage = self._get_usage_from_provider_response(chunk)

                                        stream_manager.update_usage(usage)

                                        final_response = stream_manager.get_final_response()

                                        yield stream_response, final_response
                                        return  # Stop the generator

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
        response: GenerateContentResponse,
    ) -> ChatResponseUsage:
        return ChatResponseUsage(
            total_tokens=response.usage_metadata.total_token_count,
            prompt_tokens=response.usage_metadata.prompt_token_count,
            completion_tokens=response.usage_metadata.candidates_token_count,
        )

    def parse_response(
        self,
        response: GenerateContentResponse,
    ) -> ChatResponse:
        usage, usage_charge = self.get_usage_and_charge(response)
        return ChatResponse(
            model=response.model_version,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            usage=usage,
            usage_charge=usage_charge,
            choices=[
                ChatResponseChoice(
                    index=index,
                    content=ChatResponseContentItem(
                        role=candidate.content.role,
                        text=candidate.content.parts[0].text,
                        function_call=None,
                    ),
                )
                for index, candidate in enumerate(response.candidates)
            ],
            metadata=AIModelCallResponseMetaData(
                streaming=False,
                duration_seconds=0,  # TODO
                provider_data={
                    "prompt_feedback": generic_obj_to_dict(
                        response.prompt_feedback,
                    )
                },
            ),
        )
