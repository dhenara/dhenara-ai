import logging

from google.genai.types import (
    GenerateContentConfig,
    GenerateContentResponse,
    Part,
    SafetySetting,
    Tool,
    ToolConfig,
)

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
    ChatResponseStructuredOutput,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseTextContentItemDelta,
    ChatResponseToolCall,
    ChatResponseToolCallContentItem,
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
        instructions: dict | None = None,
    ) -> AIModelCallResponse:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("inputs must be validated with `self.validate_inputs()` before api calls")

        generate_config_args = self.get_default_generate_config_args()
        generate_config = GenerateContentConfig(**generate_config_args)

        # Process instructions

        if instructions:
            if not (isinstance(instructions, dict) and "parts" in instructions.keys()):
                raise ValueError(
                    f"Invalid Instructions format. "
                    f"Instructions should be processed and passed in prompt format. Value is {instructions} "
                )

            # Some models don't support system instructions
            if any(self.model_endpoint.ai_model.model_name.startswith(model) for model in ["gemini-1.0-pro"]):
                instruction_as_prompt = instructions

                if context:
                    context.insert(0, instruction_as_prompt)
                else:
                    context = [instruction_as_prompt]
            else:
                instructions_str = instructions["parts"][0]["text"]
                generate_config.system_instruction = instructions_str

        messages = []

        # Add previous messages and current prompt
        if context:
            messages.extend(context)

        messages.append(prompt)

        # ---  Tools ---
        if self.config.tools:
            # NOTE: Google supports extra tools other than fns, so gather all fns together into function_declarations
            # --  _tools = [tool.to_google_format() for tool in self.config.tools]
            _tools = [
                Tool(
                    **{
                        "function_declarations": [
                            self.formatter.convert_function_definition(  # A bit wiered here
                                func_def=tool.function,
                                model_endpoint=self.model_endpoint,
                            )
                            for tool in self.config.tools
                        ],
                    }
                )
            ]
            generate_config.tools = _tools

        if self.config.tool_choice:
            _tool_config = self.formatter.format_tool_choice(
                tool_choice=self.config.tool_choice,
                model_endpoint=self.model_endpoint,
            )

            generate_config.tool_config = ToolConfig(**_tool_config)

        # --- Structured Output ---
        if self.config.structured_output:
            generate_config.response_mime_type = "application/json"
            generate_config.response_schema = self.formatter.format_structured_output(
                structured_output=self.config.structured_output,
                model_endpoint=self.model_endpoint,
            )

        return {
            "contents": messages,
            "generate_config": generate_config,
        }

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        response = self._client.models.generate_content(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            contents=api_call_params["contents"],
        )
        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        response = await self._client.models.generate_content(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            contents=api_call_params["contents"],
        )
        return response

    def do_streaming_api_call_sync(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        stream = self._client.models.generate_content_stream(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            contents=api_call_params["contents"],
        )
        return stream

    async def do_streaming_api_call_async(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        stream = await self._client.models.generate_content_stream(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            contents=api_call_params["contents"],
        )
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
        candidates_tokens = response.usage_metadata.candidates_token_count or 0
        thoughts_tokens = (
            (response.usage_metadata.thoughts_token_count or 0)
            if hasattr(response.usage_metadata, "thoughts_token_count")
            else 0
        )
        tool_use_tokens = (
            (response.usage_metadata.tool_use_prompt_token_count or 0)
            if hasattr(response.usage_metadata, "tool_use_prompt_token_count")
            else 0
        )

        completion_tokens = candidates_tokens + thoughts_tokens + tool_use_tokens

        return ChatResponseUsage(
            total_tokens=response.usage_metadata.total_token_count or 0,
            prompt_tokens=response.usage_metadata.prompt_token_count or 0,
            completion_tokens=completion_tokens,
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
            if hasattr(content_item, "text") and content_item.text is not None:
                _content = content_item.text

                if self.config.structured_output is not None:
                    structured_output = ChatResponseStructuredOutput.from_model_output(
                        raw_response=_content,
                        config=self.config.structured_output,
                    )
                    return ChatResponseStructuredOutputContentItem(
                        index=index,
                        role=role,
                        structured_output=structured_output,
                    )
                else:
                    return ChatResponseTextContentItem(
                        index=index,
                        role=role,
                        text=_content,
                    )

            elif hasattr(content_item, "function_call") and content_item.function_call is not None:
                content_item_dict = content_item.function_call.model_dump()

                return ChatResponseToolCallContentItem(
                    index=index,
                    role=role,
                    tool_call=ChatResponseToolCall.from_google_format(content_item_dict),
                    metadata={},
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
            if hasattr(delta, "text"):
                return ChatResponseTextContentItemDelta(
                    index=index,
                    role=role,
                    text_delta=delta.text,
                )

            # TODO: Tools Not supported in streaming yet
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
