import logging

from google.genai.types import Content, GenerateContentConfig, GenerateContentResponse, Part, SafetySetting

# Copyright 2024-2025 Dhenara Inc. All rights reserved.
from dhenara.ai.providers.google import GoogleAIClientBase
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

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


models_not_supporting_system_instructions = ["gemini-1.0-pro"]


# -----------------------------------------------------------------------------
class GoogleAIChat(GoogleAIClientBase):
    def get_api_call_params(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("inputs must be validated with `self.validate_inputs()` before api calls")

        generate_config_args = self.get_default_generate_config_args()
        generate_config = GenerateContentConfig(**generate_config_args)

        # Process instructions
        instructions_str = self.process_instructions(instructions)
        if isinstance(instructions_str, dict):
            if context:
                context.insert(0, instructions_str)
            else:
                context = [instructions_str]
        elif instructions_str and not any(
            self.model_endpoint.ai_model.model_name.startswith(model) for model in ["gemini-1.0-pro"]
        ):
            generate_config.system_instruction = instructions_str

        history = []
        if context:
            for item in context:
                if isinstance(item, dict):
                    parts = [{"text": p["text"]} for p in item["parts"]] if "parts" in item else []
                    history.append(Content(role=item["role"], parts=parts))
                else:
                    history.append(item)

        return {
            "prompt": prompt,
            "history": history,
            "generate_config": generate_config,
        }

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        chat = self._client.chats.create(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            history=api_call_params["history"],
        )
        response = chat.send_message(message=api_call_params["prompt"])
        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        chat = self._client.chats.create(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            history=api_call_params["history"],
        )
        response = await chat.send_message(message=api_call_params["prompt"])
        return response

    def do_streaming_api_call_sync(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        chat = self._client.chats.create(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            history=api_call_params["history"],
        )
        stream = chat.send_message_stream(message=api_call_params["prompt"])
        return stream

    async def do_streaming_api_call_async(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        chat = self._client.chats.create(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            history=api_call_params["history"],
        )
        stream = await chat.send_message_stream(message=api_call_params["prompt"])
        return stream

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

    def parse_stream_chunk(
        self,
        chunk: GenerateContentResponse,
    ) -> StreamingChatResponse | SSEErrorResponse | None:
        """Handle streaming response with progress tracking and final response"""

        processed_chunks = []

        self.streaming_manager.provider_metadata = None

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

            processed_chunks.append(stream_response)

            # Check if this is the final chunk
            is_done = bool(candidate.finish_reason)

            if is_done:
                usage = self._get_usage_from_provider_response(chunk)
                self.streaming_manager.update_usage(usage)

        return processed_chunks

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
            return self.get_unknown_content_type_item(
                index=index,
                role=role,
                unknown_item=content_item,
                streaming=False,
            )

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
            return self.get_unknown_content_type_item(
                index=index,
                role=role,
                unknown_item=delta,
                streaming=True,
            )
