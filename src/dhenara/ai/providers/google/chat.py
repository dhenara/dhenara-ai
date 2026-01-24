import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, cast

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
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt, SystemInstruction
from dhenara.ai.types.shared.api import SSEErrorResponse
from dhenara.ai.utils.dai_disk import DAI_JSON

logger = logging.getLogger(__name__)


models_not_supporting_system_instructions = ["gemini-1.0-pro"]


def _process_thought_signature(thought_signature: str | bytes | None) -> str | None:
    import base64

    if thought_signature is None:
        return None

    if isinstance(thought_signature, bytes):
        if not thought_signature:
            return None
        try:
            return base64.b64encode(thought_signature).decode("utf-8")
        except Exception:
            try:
                return thought_signature.decode("utf-8", errors="replace")
            except Exception:
                # return None
                pass

    return str(thought_signature)


# -----------------------------------------------------------------------------
class GoogleAIChat(GoogleAIClientBase):
    def get_api_call_params(
        self,
        prompt: str | dict | Prompt | None,
        context: Sequence[str | dict | Prompt] | None = None,
        instructions: dict[str, Any] | list[str | dict | SystemInstruction] | None = None,
        messages: Sequence[MessageItem] | None = None,
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        formatter = self.formatter
        if formatter is None:
            raise RuntimeError("Formatter not initialized")

        if self._input_validation_pending:
            raise ValueError("inputs must be validated with `self.validate_inputs()` before api calls")

        generate_config_args = self.get_default_generate_config_args()
        generate_config = GenerateContentConfig(**generate_config_args)

        context_list: list[dict[str, Any]] = []
        if context:
            ctx_items = list(context)
            context_list = [item for item in ctx_items if isinstance(item, dict)]
            if len(context_list) != len(ctx_items):
                raise ValueError(
                    "Invalid context format. Context should be processed and passed in prompt format (dict)."
                )
            # context_list already populated above

        # Process instructions

        if instructions:
            if not isinstance(instructions, dict) or "parts" not in instructions:
                raise ValueError(
                    f"Invalid Instructions format. "
                    f"Instructions should be processed and passed in prompt format. Value is {instructions} "
                )

            # Some models don't support system instructions
            if any(self.model_endpoint.ai_model.model_name.startswith(model) for model in ["gemini-1.0-pro"]):
                instruction_as_prompt = instructions

                context_list.insert(0, instruction_as_prompt)
            else:
                instructions_str = instructions["parts"][0]["text"]
                generate_config.system_instruction = instructions_str

        messages_list: list[dict[str, Any]] = []

        # Add previous messages and current prompt
        if messages is not None:
            # Convert MessageItem objects to Google format
            formatted_messages = formatter.format_messages(
                messages=messages,
                model_endpoint=self.model_endpoint,
            )
            messages_list = formatted_messages
        else:
            if context_list:
                messages_list.extend(context_list)
            if prompt is not None:
                if not isinstance(prompt, dict):
                    raise ValueError(
                        "Invalid prompt format. Prompt should be processed and passed in prompt format (dict)."
                    )
                messages_list.append(prompt)

        # ---  Tools ---
        if self.config.tools:
            # NOTE: Google supports extra tools other than fns, so gather all fns together into function_declarations
            # --  _tools = [tool.to_google_format() for tool in self.config.tools]
            function_declarations = [
                formatter.convert_function_definition(
                    func_def=tool.function,
                    model_endpoint=self.model_endpoint,
                )
                for tool in self.config.tools
            ]
            _tools = [
                Tool(
                    function_declarations=cast(Any, function_declarations),
                )
            ]
            generate_config.tools = _tools

        if self.config.tool_choice:
            _tool_config = formatter.format_tool_choice(
                tool_choice=self.config.tool_choice,
                model_endpoint=self.model_endpoint,
            )

            generate_config.tool_config = ToolConfig(**_tool_config)

        # --- Structured Output ---
        structured_output_config = self._get_structured_output_config()
        if structured_output_config:
            generate_config.response_mime_type = "application/json"
            generate_config.response_schema = formatter.format_structured_output(
                structured_output=structured_output_config,
                model_endpoint=self.model_endpoint,
            )

        return {
            "contents": messages_list,
            "generate_config": generate_config,
        }

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> GenerateContentResponse:
        client = self._client
        if client is None:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")
        response = client.models.generate_content(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            contents=api_call_params["contents"],
        )
        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> GenerateContentResponse:
        client = self._client
        if client is None:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")
        response = await client.models.generate_content(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            contents=api_call_params["contents"],
        )
        return response

    def do_streaming_api_call_sync(
        self,
        api_call_params: dict[str, Any],
    ) -> Iterator[object]:
        client = self._client
        if client is None:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")
        stream = client.models.generate_content_stream(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            contents=api_call_params["contents"],
        )
        return stream

    async def do_streaming_api_call_async(
        self,
        api_call_params: dict[str, Any],
    ) -> AsyncIterator[object]:
        client = self._client
        if client is None:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")
        stream = await client.models.generate_content_stream(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            contents=api_call_params["contents"],
        )
        return stream

    def get_default_generate_config_args(self) -> dict:
        max_output_tokens, max_reasoning_tokens = self.config.get_max_output_tokens(self.model_endpoint.ai_model)
        model_settings = self.model_endpoint.ai_model.get_settings()
        safety_settings = [
            SafetySetting(
                category=cast(Any, "HARM_CATEGORY_DANGEROUS_CONTENT"),
                threshold=cast(Any, "BLOCK_ONLY_HIGH"),
            )
        ]

        config_params: dict[str, Any] = {
            "candidate_count": 1,
            "safety_settings": safety_settings,
        }

        if max_output_tokens:
            config_params["max_output_tokens"] = max_output_tokens

        if self.config.reasoning and model_settings.supports_reasoning:
            thinking_level_supported = not self.model_endpoint.ai_model.model_name.startswith(("gemini-2", "gemini-1"))
            if thinking_level_supported:
                effort = self.config.reasoning_effort

                # To google thinking_level
                if effort in ["minimal", "low"]:
                    thinking_level = "low"
                elif effort in ["medium", "high"]:
                    thinking_level = "high"
                else:
                    logger.error(f"Illegal reasoning effort {effort}. Setting thinking_level to 'low'")
                    thinking_level = "low"

                config_params["thinking_config"] = ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=cast(Any, thinking_level),
                )
            else:
                config_params["thinking_config"] = ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=max_reasoning_tokens,
                )

        return config_params

    def parse_stream_chunk(
        self,
        chunk: object,
    ) -> StreamingChatResponse | SSEErrorResponse | list[StreamingChatResponse | SSEErrorResponse] | None:
        """Handle streaming response with progress tracking and final response"""

        if not isinstance(chunk, GenerateContentResponse):
            raise TypeError(f"Unexpected chunk type for streaming chat response: {type(chunk)}")

        processed_chunks = []

        sm = self.streaming_manager
        if sm is None:
            raise RuntimeError("Streaming manager not initialized")

        sm.provider_metadata = {}

        # Process content
        if chunk.candidates:
            choice_deltas = []
            last_finish_reason = None
            for candidate_index, candidate in enumerate(chunk.candidates):
                last_finish_reason = getattr(candidate, "finish_reason", None)
                content_deltas = []
                content = getattr(candidate, "content", None)
                if content is None:
                    continue
                parts = getattr(content, "parts", None) or []
                role = getattr(content, "role", None)
                role_str = role if isinstance(role, str) else "assistant"
                for part_index, part in enumerate(parts):
                    content_deltas.append(
                        self.process_content_item_delta(
                            index=part_index,
                            role=role_str,
                            delta=part,
                        )
                    )

                # Serialize provider-native content for diagnostics (avoid odd keys)
                try:
                    provider_content = content.model_dump() if hasattr(content, "model_dump") else str(content)
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

            response_chunk = sm.update(choice_deltas=choice_deltas)
            stream_response = StreamingChatResponse(
                id=None,  # No 'id' from google
                data=response_chunk,
            )

            processed_chunks.append(stream_response)

            # Check if this is the final chunk
            is_done = bool(last_finish_reason)

            if is_done:
                usage = self._get_usage_from_provider_response(chunk)
                sm.update_usage(usage)

                # TODO: # Investigate if Google provides a final
                # aggregated response natively either via fn or via chunks
                # and plug in into streaming_manager.native_final_response_dai/sdk

        return processed_chunks

    def _get_usage_from_provider_response(
        self,
        response: GenerateContentResponse,
    ) -> ChatResponseUsage:
        usage_md = getattr(response, "usage_metadata", None)
        if usage_md is None:
            return ChatResponseUsage(total_tokens=0, prompt_tokens=0, completion_tokens=0)

        candidates_tokens = getattr(usage_md, "candidates_token_count", 0) or 0
        thoughts_tokens = (
            (getattr(usage_md, "thoughts_token_count", 0) or 0) if hasattr(usage_md, "thoughts_token_count") else 0
        )
        tool_use_tokens = (
            (getattr(usage_md, "tool_use_prompt_token_count", 0) or 0)
            if hasattr(usage_md, "tool_use_prompt_token_count")
            else 0
        )

        completion_tokens = candidates_tokens + thoughts_tokens + tool_use_tokens

        return ChatResponseUsage(
            total_tokens=getattr(usage_md, "total_token_count", 0) or 0,
            prompt_tokens=getattr(usage_md, "prompt_token_count", 0) or 0,
            completion_tokens=completion_tokens,
            reasoning_tokens=thoughts_tokens if thoughts_tokens > 0 else None,
        )

    def parse_response(
        self,
        response: object,
    ) -> ChatResponse:
        if not isinstance(response, GenerateContentResponse):
            raise TypeError(f"Unexpected response type for chat response: {type(response)}")

        usage, usage_charge = self.get_usage_and_charge(response)

        usage_chat: ChatResponseUsage | None
        if usage is None:
            usage_chat = None
        elif isinstance(usage, ChatResponseUsage):
            usage_chat = usage
        else:
            raise TypeError(f"Unexpected usage type for chat response: {type(usage)}")

        model_name = response.model_version or self.model_name_in_api_calls
        return ChatResponse(
            model=model_name,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            usage=usage_chat,
            usage_charge=usage_charge,
            provider_response=self.serialize_provider_response(response),
            choices=[
                ChatResponseChoice(
                    index=choice_index,
                    finish_reason=candidate.finish_reason,
                    stop_sequence=None,
                    contents=GoogleMessageConverter.provider_message_to_dai_content_items(
                        message=candidate.content,
                        structured_output_config=self._get_structured_output_config(),
                    ),
                    metadata={},  # Choice metadata
                )
                for choice_index, candidate in enumerate(response.candidates or [])
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
        delta: object,
    ) -> ChatResponseContentItemDelta:
        # Accept both typed SDK objects and dict-like parts
        try:

            def _try_model_dump(obj: object) -> dict[str, Any]:
                if hasattr(obj, "model_dump"):
                    try:
                        dumped = cast(Any, obj).model_dump()
                        if isinstance(dumped, dict):
                            return cast(dict[str, Any], dumped)
                    except Exception:
                        pass
                return {}

            def _get_attr(obj: object, attr: str, default: Any = None) -> Any:
                if not hasattr(obj, "__dict__") and hasattr(obj, "get"):
                    return cast(Any, obj).get(attr, default)
                return getattr(obj, attr, default)

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
                    name = _get_attr(fn, "name", None)
                    args = _get_attr(fn, "args", None)
                    # Capture thought_signature from this delta or any previously stashed one
                    thought_signature = _get_attr(delta, "thought_signature", None)
                    if not thought_signature:
                        thought_signature = getattr(self, "_pending_thought_signature", None)

                    from dhenara.ai.types.genai import ChatResponseToolCall as _Tool

                    parsed = _Tool.parse_args_str_or_dict(args or {})
                    tool_call = ChatResponseToolCall(
                        call_id=None,  # Google often omits IDs in streaming
                        id=None,
                        name=str(name or ""),
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
                            "part": _try_model_dump(delta),
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

                return ChatResponseTextContentItemDelta(
                    index=index,
                    role=role,
                    text_delta=DAI_JSON.dumps(resp) if resp is not None else "",
                )

            # Fallback: generic
            return ChatResponseGenericContentItemDelta(
                index=index,
                role=role,
                metadata={"part": _try_model_dump(delta)},
            )
        except Exception:
            return self.get_unknown_content_type_item(
                index=index,
                role=role,
                unknown_item=delta,
                streaming=True,
            )
