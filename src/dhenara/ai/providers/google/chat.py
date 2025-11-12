import logging

from google.genai.types import (
    GenerateContentConfig,
    GenerateContentResponse,
    SafetySetting,
    ThinkingConfig,
    Tool,
    ToolConfig,
)

# Copyright 2024-2025 Dhenara Inc. All rights reserved.
from dhenara.ai.providers.google import GoogleAIClientBase
from dhenara.ai.providers.google.message_converter import GoogleMessageConverter
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseContentItemDelta,
    ChatResponseGenericContentItemDelta,
    ChatResponseReasoningContentItemDelta,
    ChatResponseTextContentItemDelta,
    ChatResponseToolCall,
    ChatResponseToolCallContentItemDelta,
    ChatResponseUsage,
    StreamingChatResponse,
)
from dhenara.ai.types.shared.api import SSEErrorResponse

logger = logging.getLogger(__name__)


models_not_supporting_system_instructions = ["gemini-1.0-pro"]


def _process_thought_signature(thought_signature: str | bytes | None) -> str | None:
    import base64

    if thought_signature and not isinstance(thought_signature, str):
        try:
            return base64.b64encode(thought_signature).decode("utf-8")
        except Exception:
            # Keep as-is if encoding fails
            pass

    return thought_signature


# -----------------------------------------------------------------------------
class GoogleAIChat(GoogleAIClientBase):
    def get_api_call_params(
        self,
        prompt: dict | None,
        context: list[dict] | None = None,
        instructions: dict | None = None,
        messages: list[dict] | None = None,
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

        messages_list = []

        # Add previous messages and current prompt
        if messages is not None:
            # Convert MessageItem objects to Google format
            formatted_messages = self.formatter.format_messages(
                messages=messages,
                model_endpoint=self.model_endpoint,
            )
            messages_list = formatted_messages
        else:
            if context:
                messages_list.extend(context)
            if prompt is not None:
                messages_list.append(prompt)

        # ---  Tools ---
        if self.config.tools:
            # NOTE: Google supports extra tools other than fns, so gather all fns together into function_declarations
            # --  _tools = [tool.to_google_format() for tool in self.config.tools]
            function_declarations = [
                self.formatter.convert_function_definition(
                    func_def=tool.function,
                    model_endpoint=self.model_endpoint,
                )
                for tool in self.config.tools
            ]
            _tools = [
                Tool(
                    function_declarations=function_declarations,
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
            "contents": messages_list,
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

        if max_reasoning_tokens:
            config_params["thinking_config"] = ThinkingConfig(
                include_thoughts=True,
                thinking_budget=max_reasoning_tokens,
            )

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
                for part_index, part in enumerate(candidate.content.parts or []):
                    content_deltas.append(
                        self.process_content_item_delta(
                            index=part_index,
                            role=candidate.content.role,
                            delta=part,
                        )
                    )

                # Serialize provider-native content for diagnostics (avoid odd keys)
                try:
                    provider_content = (
                        candidate.content.model_dump()
                        if hasattr(candidate.content, "model_dump")
                        else str(candidate.content)
                    )
                except Exception:
                    provider_content = None

                choice_deltas.append(
                    ChatResponseChoiceDelta(
                        index=candidate_index,
                        finish_reason=candidate.finish_reason,
                        stop_sequence=None,
                        content_deltas=content_deltas,
                        metadata={
                            "safety_ratings": candidate.safety_ratings,
                            "provider_content": provider_content,
                        },  # Choice metadata
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

                # TODO: # Investigate if Google provides a final
                # aggregated response natively either via fn or via chunks
                # and plug in into streaming_manager.native_final_response_dai/sdk

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
            reasoning_tokens=thoughts_tokens if thoughts_tokens > 0 else None,
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
            provider_response=self.serialize_provider_response(response),
            choices=[
                ChatResponseChoice(
                    index=choice_index,
                    finish_reason=candidate.finish_reason,
                    stop_sequence=None,
                    contents=GoogleMessageConverter.provider_message_to_dai_content_items(
                        message=candidate.content,
                        structured_output_config=self.config.structured_output,
                    ),
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

    # Streaming
    def process_content_item_delta(
        self,
        index: int,
        role: str,
        delta,
    ) -> ChatResponseContentItemDelta:
        # Accept both typed SDK objects and dict-like parts
        try:

            def _get_attr(obj, attr, default=None):
                return getattr(obj, attr, default)

            # If mapping-like, switch to dict getter
            if not hasattr(delta, "__dict__") and hasattr(delta, "get"):

                def _get_attr(obj, attr, default=None):
                    return obj.get(attr, default)

            # Reasoning/thought text: stream as text_delta into message_contents (type="thinking")
            if _get_attr(delta, "thought", default=False) is True:
                delta_text = _get_attr(delta, "text", None)
                thought_signature = _get_attr(delta, "thought_signature", None)
                # Stash signature for upcoming function_call if present
                try:
                    if thought_signature:
                        self._pending_thought_signature = thought_signature
                except Exception:
                    pass
                return ChatResponseReasoningContentItemDelta(
                    index=index,
                    role=role,
                    text_delta=delta_text,  # append into reasoning.message_contents (thinking)
                    thinking_summary_delta=None,
                    thinking_signature=_process_thought_signature(thought_signature),
                )

            # Text piece
            if _get_attr(delta, "text", None) is not None:
                return ChatResponseTextContentItemDelta(
                    index=index,
                    role=role,
                    text_delta=_get_attr(delta, "text", None),
                )

            # Function/tool call
            fn = _get_attr(delta, "function_call", None)
            if fn is not None:
                try:
                    # Normalize function object (SDK or dict)
                    name = _get_attr(fn, "name", None) if hasattr(fn, "__dict__") else fn.get("name")
                    args = _get_attr(fn, "args", None) if hasattr(fn, "__dict__") else fn.get("args")
                    # Capture thought_signature from this delta or any previously stashed one
                    thought_signature = _get_attr(delta, "thought_signature", None)
                    if not thought_signature:
                        thought_signature = getattr(self, "_pending_thought_signature", None)

                    from dhenara.ai.types.genai import ChatResponseToolCall as _Tool

                    parsed = _Tool.parse_args_str_or_dict(args)
                    tool_call = ChatResponseToolCall(
                        call_id=None,  # Google often omits IDs in streaming
                        id=None,
                        name=name,
                        arguments=parsed.get("arguments_dict") or {},
                        raw_data=parsed.get("raw_data"),
                        parse_error=parsed.get("parse_error"),
                    )
                    delta_obj = ChatResponseToolCallContentItemDelta(
                        index=index,
                        role=role,
                        tool_call=tool_call,
                        metadata={
                            "google_function_call": True,
                            **(
                                {"thought_signature": _process_thought_signature(thought_signature)}
                                if thought_signature
                                else {}
                            ),
                        },
                    )
                    # Clear the stashed signature after attaching to the call
                    try:
                        if getattr(self, "_pending_thought_signature", None):
                            self._pending_thought_signature = None
                    except Exception:
                        pass
                    return delta_obj
                except Exception as e:
                    return ChatResponseGenericContentItemDelta(
                        index=index,
                        role=role,
                        metadata={
                            "part": getattr(delta, "model_dump", lambda: {})(),
                            "error": str(e),
                        },
                    )

            # Function response
            fn_resp = _get_attr(delta, "function_response", None)
            if fn_resp is not None:
                # Represent function responses as text deltas with JSON body for now
                try:
                    resp = (
                        _get_attr(fn_resp, "response", None)
                        if hasattr(fn_resp, "__dict__")
                        else fn_resp.get("response")
                    )
                except Exception:
                    resp = None
                import json as _json

                return ChatResponseTextContentItemDelta(
                    index=index,
                    role=role,
                    text_delta=_json.dumps(resp) if resp is not None else "",
                )

            # Fallback: generic
            return ChatResponseGenericContentItemDelta(
                index=index,
                role=role,
                metadata={"part": getattr(delta, "model_dump", lambda: {})()},
            )
        except Exception:
            return self.get_unknown_content_type_item(
                index=index,
                role=role,
                unknown_item=delta,
                streaming=True,
            )
