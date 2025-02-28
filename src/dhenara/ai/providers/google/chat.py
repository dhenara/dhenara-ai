import logging

# Copyright 2024-2025 Dhenara Inc. All rights reserved.
from collections.abc import AsyncGenerator

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
    ChatResponseChoiceDelta,
    ChatResponseContentItem,
    ChatResponseContentItemDelta,
    ChatResponseGenericContentItem,
    ChatResponseGenericContentItemDelta,
    ChatResponseTextContentItem,
    ChatResponseTextContentItemDelta,
    ChatResponseUsage,
    StreamingChatResponse,
)
from dhenara.ai.types.shared.api import SSEErrorResponse
from google.genai.types import GenerateContentConfig, GenerateContentResponse, Part, SafetySetting

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
        max_output_tokens, max_reasoning_tokens = self.config.get_max_output_tokens(self.model_endpoint.ai_model)
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

        if max_output_tokens:
            config_params["max_output_tokens"] = max_output_tokens

        return config_params

    async def _handle_streaming_response(
        self,
        stream: AsyncGenerator[GenerateContentResponse],
    ) -> AsyncGenerator[tuple[StreamingChatResponse | SSEErrorResponse, AIModelCallResponse | None]]:
        self.streaming_manager = StreamingManager(
            model_endpoint=self.model_endpoint,
        )

        try:
            async for chunk in stream:
                stream_response: StreamingChatResponse | None = None
                final_response: AIModelCallResponse | None = None
                provider_metadata = None

                # Process content
                if chunk.candidates:
                    choice_deltas = []
                    for candidate_index, candidate in enumerate(chunk.candidates):
                        content_deltas = []
                        for part_index, part in enumerate(candidate.content.parts):
                            content_deltas.append(
                                self.process_content_item_delta(
                                    index=part_index,
                                    role=candidate.content.role,
                                    delta=part,
                                )
                            )
                        choice_deltas.append(
                            ChatResponseChoiceDelta(
                                index=candidate_index,
                                finish_reason=candidate.finish_reason,
                                stop_sequence=None,
                                content_deltas=content_deltas,
                                metadata={"safety_ratings": candidate.safety_ratings, "": candidate.content},  # Choice metadata
                            )
                        )

                    response_chunk = self.streaming_manager.update(choice_deltas=choice_deltas)
                    stream_response = StreamingChatResponse(
                        id=None,  # No 'id' from google
                        data=response_chunk,
                    )

                    yield stream_response, final_response

                    # Check if this is the final chunk
                    is_done = bool(candidate.finish_reason)

                    if is_done:
                        usage = self._get_usage_from_provider_response(chunk)
                        self.streaming_manager.update_usage(usage)

            # API has stopped streaming, get final response
            logger.debug("API has stopped streaming, processsing final response")
            final_response = self.streaming_manager.complete(provider_metadata=provider_metadata)

            yield None, final_response
            return  # Stop the generator
        except Exception as e:
            error_response = self._create_streaming_error_response(exc=e)
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
                    index=choice_index,
                    finish_reason=candidate.finish_reason,
                    stop_sequence=None,
                    contents=[
                        self.process_content_item(
                            index=part_index,
                            role=candidate.content.role,
                            content_item=part,
                        )
                        for part_index, part in enumerate(candidate.content.parts)
                    ],
                    metadata={},  # Choice metadata
                )
                for choice_index, candidate in enumerate(response.candidates)
            ],
            metadata=AIModelCallResponseMetaData(
                streaming=False,
                duration_seconds=0,  # TODO
                provider_metadata={},
            ),
        )

    def process_content_item(
        self,
        index: int,
        role: str,
        content_item: Part,
    ) -> ChatResponseContentItem:
        if isinstance(content_item, Part):
            if content_item.text:
                return ChatResponseTextContentItem(
                    index=index,
                    role=role,
                    text=content_item.text,
                )
            else:
                return ChatResponseGenericContentItem(
                    index=index,
                    role=role,
                    metadata={"part": content_item.model_dump()},
                )
        else:
            return self.get_unknown_content_type_item(role=role, unknown_item=content_item, streaming=False)

    # Streaming
    def process_content_item_delta(
        self,
        index: int,
        role: str,
        delta,
    ) -> ChatResponseContentItemDelta:
        if isinstance(delta, Part):
            if delta.text:
                return ChatResponseTextContentItemDelta(
                    index=index,
                    role=role,
                    text_delta=delta.text,
                )
            else:
                return ChatResponseGenericContentItemDelta(
                    index=index,
                    role=role,
                    metadata={"part": delta.model_dump()},
                )

        else:
            return self.get_unknown_content_type_item(role=role, unknown_item=delta, streaming=True)
