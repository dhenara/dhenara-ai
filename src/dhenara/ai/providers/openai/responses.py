import json
import logging
from typing import Any

from dhenara.ai.providers.openai import OpenAIClientBase
from dhenara.ai.providers.openai.formatter import OpenAIFormatter
from dhenara.ai.providers.openai.message_converter import OpenAIMessageConverter
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseContentItem,
    ChatResponseReasoningContentItemDelta,
    ChatResponseTextContentItemDelta,
    ChatResponseToolCall,
    ChatResponseToolCallContentItemDelta,
    ChatResponseUsage,
    StreamingChatResponse,
)
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse

logger = logging.getLogger(__name__)


class OpenAIResponses(OpenAIClientBase):
    """OpenAI Responses API client (text + tools + structured output).

    Phase 1 scope:
    - OpenAI provider only (not Azure variants)
    - Text and tools, structured output, reasoning, streaming events (text deltas)
    - Vision inputs supported via input_image (data URL) when files are provided
    - Image generation remains in image.py (legacy endpoint)
    """

    # ----------------------- Request build -----------------------
    def _build_responses_input(
        self,
        *,
        prompt: dict | None,
        context: list[dict] | None,
        messages: list | None,
    ) -> list[dict]:
        """Build the Responses API 'input' array of role/content items."""
        input_items: list[dict] = []

        if messages is not None:
            # Convert Dhenara messages to Responses input messages
            for mi in messages:
                converted = OpenAIFormatter.convert_dai_message_item_to_provider(
                    message_item=mi,
                    model_endpoint=self.model_endpoint,
                )
                if isinstance(converted, list):
                    input_items.extend(converted)
                elif converted:
                    input_items.append(converted)
        else:
            if context is not None:
                input_items.extend(context)
            if prompt is not None:
                input_items.append(prompt)

        return input_items

    def get_api_call_params(
        self,
        prompt: dict | None,
        context: list[dict] | None = None,
        instructions: dict | None = None,
        messages: list | None = None,
    ) -> AIModelCallResponse:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("inputs must be validated with `self.validate_inputs()` before api calls")

        api = self.model_endpoint.api
        if api.provider != AIModelAPIProviderEnum.OPEN_AI:
            raise ValueError("OpenAIResponses only supports AIModelAPIProviderEnum.OPEN_AI in Phase 1")

        input_messages = self._build_responses_input(
            prompt=prompt,
            context=context or [],
            messages=messages,
        )

        # Base args
        args: dict[str, Any] = {
            "model": self.model_name_in_api_calls,
            "input": input_messages,
            "stream": self.config.streaming,
        }
        if instructions:
            # Instruction at this point are in Dhenara format
            instructions_text = instructions.get("content")
            args["instructions"] = instructions_text

        # Max tokens (Responses uses max_output_tokens)
        # Note: Responses API doesn't have a separate max_reasoning_tokens parameter.
        # Reasoning tokens come out of max_output_tokens budget.
        max_output_tokens, max_reasoning_tokens = self.config.get_max_output_tokens(self.model_endpoint.ai_model)
        if max_output_tokens is not None:
            args["max_output_tokens"] = max_output_tokens

        # Reasoning configuration
        # Responses API: reasoning = {effort: "low"|"medium"|"high",
        #                             generate_summary: "auto"|"concise"|"detailed"}
        # Note: Unlike Anthropic's budget_tokens, OpenAI uses max_output_tokens for
        # total (text + reasoning)
        if self.config.reasoning:
            reasoning_config: dict[str, Any] = {}

            # Effort level
            if self.config.reasoning_effort is not None:
                effort = self.config.reasoning_effort
                # Normalize Dhenara "minimal" -> OpenAI "low"
                if isinstance(effort, str) and effort.lower() == "minimal":
                    effort = "low"
                reasoning_config["effort"] = effort
            else:
                # Default effort if reasoning is enabled but not specified
                reasoning_config["effort"] = "low"

            # Inorder to get the reasoning text, OpenAI need to pass `summary` as any of the
            # : "auto", "concise", "detailed"
            reasoning_config["summary"] = "detailed"

            if reasoning_config:
                args["reasoning"] = reasoning_config

        # Log warning about reasoning token budget (informational)
        if max_reasoning_tokens is not None:
            logger.debug(
                f"Responses API: max_reasoning_tokens ({max_reasoning_tokens}) is advisory only. "
                f"Reasoning tokens come from max_output_tokens budget ({max_output_tokens})."
            )

        # Tools and tool choice
        if self.config.tools:
            # Use Responses-specific tool schema (top-level name)
            tools_formatted = None
            try:
                tools_formatted = self.formatter.format_tools(
                    tools=self.config.tools,
                    model_endpoint=self.model_endpoint,
                )
            except Exception:
                logger.exception("Error formatting tools for Responses API")

            if tools_formatted:
                args["tools"] = tools_formatted
        if self.config.tool_choice:
            args["tool_choice"] = self.formatter.format_tool_choice(
                tool_choice=self.config.tool_choice,
                model_endpoint=self.model_endpoint,
            )

        if self.config.structured_output:
            # INFO: Do not remove this comment block

            # # Structured output for responses API had buidl in pyd support along with a dedicated parsing via
            # `response = self._client.responses.parse(**args)`
            # But this will FAIL if the pyd mdoel is complex or has nested models or even defined outside of
            # the current module it seems.
            # Thus, we always use JSON schema via text.format instead.
            #
            #
            # if not self.config.streaming:
            #     schema_pyd_model = self.config.structured_output.model_class_reference
            #     args["text_format"] = schema_pyd_model

            # Always use JSON schema via text.format for structured output.
            # This avoids SDK-specific pydantic integration differences and lets us enforce strict schemas.
            schema_dict = self.formatter.convert_structured_output(
                structured_output=self.config.structured_output,
                model_endpoint=self.model_endpoint,
            )
            # Extract json_schema from the formatted structure
            if schema_dict.get("type") == "json_schema" and "json_schema" in schema_dict:
                json_schema_info = schema_dict["json_schema"]
                args["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema_info.get("name", "output"),
                        "schema": json_schema_info.get("schema", {}),
                        "strict": json_schema_info.get("strict", True),
                    }
                }
                if "description" in json_schema_info:
                    args["text"]["format"]["description"] = json_schema_info["description"]

        # Metadata: attach user id if available
        user = self.config.get_user()
        if user:
            args["metadata"] = {"user_id": user}

        # Streaming options
        # Note: Some SDK versions of Responses API do not support stream_options.include_usage.
        # We'll omit stream_options to avoid 400 errors and rely on usage in the final response.
        # if self.config.streaming:
        #     args["stream_options"] = {"include_usage": True}

        # Extra options passthrough (allow overrides)
        if self.config.options:
            args.update(self.config.options)

        return {"response_args": args}

    # ----------------------- API calls -----------------------

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        args = dict(api_call_params["response_args"])
        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError("OpenAIResponses doens't supports AIModelAPIProviderEnum.MICROSOFT_AZURE_AI in Phase 1")
        else:
            # Use create() for both streaming and non-streaming.
            # We always pass text.format when structured output is requested.
            response = self._client.responses.create(**args)
        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        args = dict(api_call_params["response_args"])
        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError("OpenAIResponses doens't supports AIModelAPIProviderEnum.MICROSOFT_AZURE_AI in Phase 1")
        else:
            # Use create() for both streaming and non-streaming.
            # We always pass text.format when structured output is requested.
            response = await self._client.responses.create(**args)
        return response

    def do_streaming_api_call_sync(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        args = dict(api_call_params["response_args"])
        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError("OpenAIResponses doens't supports AIModelAPIProviderEnum.MICROSOFT_AZURE_AI in Phase 1")
        else:
            # Never use parse() for streaming calls; rely on text-based fallback
            stream = self._client.responses.create(**args)

        return stream

    async def do_streaming_api_call_async(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        args = dict(api_call_params["response_args"])
        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError("OpenAIResponses doens't supports AIModelAPIProviderEnum.MICROSOFT_AZURE_AI in Phase 1")
        else:
            # Never use parse() for streaming calls; rely on text-based fallback
            stream = await self._client.responses.create(**args)

        return stream

    # ----------------------- Parsing -----------------------
    def _get_usage_from_provider_response(self, response) -> ChatResponseUsage | None:
        try:
            usage = getattr(response, "usage", None)
            if not usage:
                return None
            # Responses usage typically has input_tokens and output_tokens
            total = getattr(usage, "total_tokens", None)
            prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)

            # Extract reasoning tokens from output_tokens_details
            reasoning_tokens = None
            output_details = getattr(usage, "output_tokens_details", None)
            if output_details:
                reasoning_tokens = getattr(output_details, "reasoning_tokens", None)

            if total is None and prompt is not None and completion is not None:
                total = int(prompt) + int(completion)

            return ChatResponseUsage(
                total_tokens=total,
                prompt_tokens=prompt,
                completion_tokens=completion,
                reasoning_tokens=reasoning_tokens,
            )
        except Exception as e:
            logger.debug(f"_get_usage_from_provider_response (Responses): {e}")
            return None

    def _parse_tool_arguments(self, arguments: str | dict) -> dict:
        """Parse tool call arguments from string or dict."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except Exception:
                logger.debug(f"Failed to parse tool arguments as JSON: {arguments}")
                return {}
        return {}

    def parse_response(self, response) -> ChatResponse:
        """Parse OpenAI Responses API response into Dhenara ChatResponse.

        Response structure:
        - response.output: list of output items (message, reasoning, function_call, etc.)
        - response.reasoning: reasoning config/summary (root level)
        - response.output_text: convenience field (text from first message)

        Output item types:
        - ResponseOutputMessage (type='message'): contains content list with ResponseOutputText items
        - ResponseReasoningItem (type='reasoning'): thinking/reasoning block
        - ResponseFunctionToolCall (type='function_call'): tool call with name/arguments/call_id
        """
        model = getattr(response, "model", None) or self.model_endpoint.ai_model.model_name
        contents: list[ChatResponseContentItem] = []
        content_index = 0

        # Parse output array for all content types
        output = getattr(response, "output", None) or []

        # Check for incomplete response (reasoning models may use all tokens for thinking)
        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        if status == "incomplete" and incomplete_details:
            reason = getattr(incomplete_details, "reason", None)
            logger.warning(f"Incomplete response: reason={reason}")

        # for item in output:
        #    converted = OpenAIMessageConverter.provider_message_item_to_dai_content_item(
        #        message_item=item,
        #        role="assistant",
        #        index_start=content_index,
        #        ai_model_provider=self.model_endpoint.ai_model.provider,
        #        structured_output_config=self.config.structured_output,
        #    )
        #    if not converted:
        #        continue
        #    for ci in converted:
        #        # Normalize incremental indices
        #        ci.index = content_index
        #        contents.append(ci)
        #        content_index += 1

        contents = OpenAIMessageConverter.provider_message_to_dai_content_items(
            message=output,
            role="assistant",
            index_start=content_index,
            ai_model_provider=self.model_endpoint.ai_model.provider,
            structured_output_config=self.config.structured_output,
        )

        usage, usage_charge = self.get_usage_and_charge(response)

        choice = ChatResponseChoice(
            index=0,
            finish_reason=None,
            stop_sequence=None,
            contents=contents,
            metadata={},
        )

        return ChatResponse(
            model=model,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            usage=usage,
            usage_charge=usage_charge,
            choices=[choice],
            provider_response=self.serialize_provider_response(response),
            metadata=AIModelCallResponseMetaData(
                streaming=False,
                duration_seconds=0,
                provider_metadata={
                    "id": getattr(response, "id", None),
                    "created": str(getattr(response, "created", "")),
                    "object": getattr(response, "object", None),
                },
            ),
        )

    # Streaming handlers: convert Responses events to StreamingChatResponse
    def parse_stream_chunk(self, chunk) -> StreamingChatResponse | SSEErrorResponse | list | None:
        """Parse a single OpenAI Responses API streaming event into internal streaming objects.

        Returns a list of StreamingChatResponse (zero or more), a list containing SSEErrorResponse
        for error events, or None for ignorable events.
        """
        processed: list[StreamingChatResponse] = []

        # Provider metadata (grab once)
        if not self.streaming_manager.provider_metadata:
            self.streaming_manager.provider_metadata = {
                "id": getattr(chunk, "id", None),
                "created": str(getattr(chunk, "created", "")),
                "object": getattr(chunk, "object", None),
                "system_fingerprint": getattr(chunk, "system_fingerprint", None),
            }

        # Usage snapshot (if any)
        # Noramay present in chunk.usage but for events like response.completed, usage is in chunk.response.usage
        usage = getattr(chunk, "usage", None) or getattr(getattr(chunk, "response", None), "usage", None)
        if usage:
            try:
                output_details = getattr(usage, "output_tokens_details", None)
                reasoning_tokens = getattr(output_details, "reasoning_tokens", None) if output_details else None
                usage_obj = ChatResponseUsage(
                    total_tokens=getattr(usage, "total_tokens", None),
                    prompt_tokens=getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", None),
                    reasoning_tokens=reasoning_tokens,
                )
                self.streaming_manager.update_usage(usage_obj)
            except Exception as _e:
                logger.warning(f"oai_stream_chunk: failed usage parse: {_e}")

        # Event type dispatch
        event_type = getattr(chunk, "type", None)
        if not event_type:
            logger.debug(f"oai_stream_chunk: missing event_type; ignoring {chunk}")
            return None

        logger.debug(f"oai_stream_chunk: processing type={event_type}")

        # 1) Text deltas
        if event_type == "response.output_text.delta":
            delta_text = getattr(chunk, "delta", None)
            output_index = getattr(chunk, "output_index", None)
            if delta_text:
                # Check if we have a pending message ID for this output_index
                message_id = None
                message_contents = None
                if hasattr(self.streaming_manager, "pending_message_ids"):
                    message_id = self.streaming_manager.pending_message_ids.get(output_index or 0)
                if hasattr(self.streaming_manager, "pending_message_content"):
                    message_contents = self.streaming_manager.pending_message_content.get(output_index or 0)

                content_delta = ChatResponseTextContentItemDelta(
                    index=0,
                    role="assistant",
                    text_delta=delta_text,
                    message_id=message_id,
                    # Only pass message_contents when we have a valid message_id snapshot
                    message_contents=message_contents if message_id else None,
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        # 2) Reasoning deltas
        # Note: reasoning has its own dedicated text stream separate from reasoning_summary
        elif event_type == "response.reasoning_text.delta":
            delta_text = getattr(chunk, "delta", None)
            item_id = getattr(chunk, "item_id", None)
            content_index = getattr(chunk, "content_index", None)
            output_index = getattr(chunk, "output_index", None)
            if delta_text:
                # Use captured item_id if available, or fall back to chunk's item_id
                if not item_id and hasattr(self.streaming_manager, "oai_pending_reasoning_ids"):
                    item_id = self.streaming_manager.oai_pending_reasoning_ids.get(output_index or 0)
                content_delta = ChatResponseReasoningContentItemDelta(
                    index=0,
                    role="assistant",
                    text_delta=delta_text,
                    thinking_summary_delta=None,
                    thinking_id=item_id,
                    metadata={"output_index": output_index, "content_index": content_index},
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        # Reasoning summary delta (condensed reasoning for o3-mini)
        elif event_type == "response.reasoning_summary_text.delta":
            delta_text = getattr(chunk, "delta", None)
            item_id = getattr(chunk, "item_id", None)
            summary_index = getattr(chunk, "summary_index", None)
            output_index = getattr(chunk, "output_index", None)
            if delta_text:
                # Use captured item_id if available
                if not item_id and hasattr(self.streaming_manager, "oai_pending_reasoning_ids"):
                    item_id = self.streaming_manager.oai_pending_reasoning_ids.get(output_index or 0)
                content_delta = ChatResponseReasoningContentItemDelta(
                    index=0,
                    role="assistant",
                    text_delta=None,
                    thinking_summary_delta=delta_text,
                    thinking_id=item_id,
                    metadata={"output_index": output_index, "summary_index": summary_index},
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        # 3) Tool call streaming (name + arguments)
        elif event_type == "response.function_call_arguments.delta":
            # Incremental tool call arguments (JSON string pieces)
            delta_text = getattr(chunk, "delta", None)
            output_index = getattr(chunk, "output_index", None)
            # Pass along any known tool call id/name captured earlier for this output index
            meta: dict = {}
            if hasattr(self.streaming_manager, "pending_tool_ids"):
                _ids = self.streaming_manager.pending_tool_ids.get(output_index or 0)
                if _ids:
                    meta.update(_ids)
            if delta_text:
                content_delta = ChatResponseToolCallContentItemDelta(
                    index=0,
                    role="assistant",
                    arguments_delta=delta_text,
                    metadata=meta,
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        elif event_type == "response.function_call.name.delta":
            # Function call name delta (rare, but some SDKs may emit name separately)
            name_piece = getattr(chunk, "delta", None)
            output_index = getattr(chunk, "output_index", None)
            meta: dict = {"tool_name_delta": name_piece, "name": name_piece}
            if hasattr(self.streaming_manager, "pending_tool_ids"):
                _ids = self.streaming_manager.pending_tool_ids.get(output_index or 0)
                if _ids:
                    meta.update(_ids)
            if name_piece:
                # Emit a start delta with partial/known name in metadata to seed tool call
                content_delta = ChatResponseToolCallContentItemDelta(
                    index=0,
                    role="assistant",
                    metadata=meta,
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        elif event_type == "response.function_call_arguments.done":
            # Function call completed with full arguments and name
            name = getattr(chunk, "name", None)
            args_str = getattr(chunk, "arguments", None)
            item_id = getattr(chunk, "item_id", None)
            call_id = getattr(chunk, "call_id", None)
            tool_call_obj = None
            if name:
                try:
                    parsed_args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    parsed_args = {"raw": args_str}
                tool_call_obj = ChatResponseToolCall(
                    call_id=call_id,
                    id=item_id,
                    name=name,
                    arguments=parsed_args or {},
                )

            content_delta = ChatResponseToolCallContentItemDelta(
                index=0,
                role="assistant",
                tool_call=tool_call_obj,
                metadata={"call_id": call_id, "item_id": item_id, "name": name, "finalize_tool_call": True},
            )
            choice_delta = ChatResponseChoiceDelta(
                index=0,
                finish_reason=None,
                stop_sequence=None,
                content_deltas=[content_delta],
                metadata={},
            )
            response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
            processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        # 4) Lifecycle / markers
        elif event_type in ("response.text.done", "response.output_text.done"):
            pass

        elif event_type in ("response.output_item.added", "response.content_part.added"):
            # Capture message IDs and reasoning IDs for proper round-tripping
            item = getattr(chunk, "item", None)
            if item:
                item_type = getattr(item, "type", None)
                item_id = getattr(item, "id", None)
                output_index = getattr(chunk, "output_index", None)
                if item_type == "message" and item_id:
                    # Store message ID in streaming manager for later retrieval
                    if not hasattr(self.streaming_manager, "pending_message_ids"):
                        self.streaming_manager.pending_message_ids = {}
                    self.streaming_manager.pending_message_ids[output_index or 0] = item_id

                    # Also capture the content array structure if available
                    content = getattr(item, "content", None)
                    if content:
                        if not hasattr(self.streaming_manager, "pending_message_content"):
                            self.streaming_manager.pending_message_content = {}
                        # Convert to dicts for storage
                        content_array = []
                        for part in content:
                            if hasattr(part, "model_dump"):
                                content_array.append(part.model_dump())
                            elif isinstance(part, dict):
                                content_array.append(part)
                        self.streaming_manager.pending_message_content[output_index or 0] = content_array
                elif item_type == "reasoning" and item_id:
                    # Store reasoning ID for potential association with reasoning deltas
                    if not hasattr(self.streaming_manager, "oai_pending_reasoning_ids"):
                        self.streaming_manager.oai_pending_reasoning_ids = {}
                    self.streaming_manager.oai_pending_reasoning_ids[output_index or 0] = item_id

                    # CRITICAL FIX: Initialize reasoning content item when reasoning output_item is added
                    # Without this, reasoning deltas arrive but have no content item to attach to
                    # This matches the behavior in non-streaming where reasoning item is created upfront
                    content_delta = ChatResponseReasoningContentItemDelta(
                        index=0,
                        role="assistant",
                        text_delta=None,  # No text yet, just initializing the item
                        thinking_summary_delta=None,
                        thinking_id=item_id,
                        metadata={"output_index": output_index},
                    )
                    choice_delta = ChatResponseChoiceDelta(
                        index=0,
                        finish_reason=None,
                        stop_sequence=None,
                        content_deltas=[content_delta],
                        metadata={},
                    )
                    response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                    processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))
                elif item_type in ("function_call", "custom_tool_call"):
                    # Store tool call identifiers (call_id may not be available yet)
                    if not hasattr(self.streaming_manager, "pending_tool_ids"):
                        self.streaming_manager.pending_tool_ids = {}
                    # Some SDKs place name/call_id on item, else None
                    name = getattr(item, "name", None)
                    call_id = getattr(item, "call_id", None)
                    self.streaming_manager.pending_tool_ids[output_index or 0] = {
                        "item_id": item_id,
                        "call_id": call_id,
                        "name": name,
                    }

        elif event_type in ("response.output_item.done", "response.content_part.done"):
            pass

        elif event_type in ("response.reasoning_summary_part.added", "response.reasoning_summary_part.done"):
            # Reasoning summary part events (piecewise summary text)
            part = getattr(chunk, "part", None)
            text = getattr(part, "text", None) if part is not None else None
            item_id = getattr(chunk, "item_id", None)
            output_index = getattr(chunk, "output_index", None)
            summary_index = getattr(chunk, "summary_index", None)
            if text:
                content_delta = ChatResponseReasoningContentItemDelta(
                    index=0,
                    role="assistant",
                    thinking_summary_delta=text,
                    thinking_id=item_id,
                    metadata={"output_index": output_index, "summary_index": summary_index},
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        elif event_type == "response.refusal.delta":
            refusal_text = getattr(chunk, "delta", None)
            if refusal_text:
                content_delta = ChatResponseTextContentItemDelta(
                    index=0,
                    role="assistant",
                    text_delta=refusal_text,
                    metadata={"refusal": True},
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        elif event_type == "response.completed":
            # If the SDK supplies a full aggregated response here, convert it and store an
            # override so StreamingManager can return it directly.
            try:
                final_resp = getattr(chunk, "response", None)
                if final_resp is not None:
                    # Convert provider-native response using existing parser
                    final_chat = self.parse_response(final_resp)
                    self.streaming_manager.native_final_response_sdk = final_resp
                    self.streaming_manager.native_final_response_dai = final_chat
                else:
                    logger.error("oai_stream_chunk: response.completed has no final response object")
            except Exception as _e:
                logger.error(f"oai_stream_chunk: unable to attach final aggregated response: {_e}")

            processed.append(self.streaming_manager.get_streaming_done_chunk())

        # 5) Error
        elif event_type == "error":
            # Emit a proper SSEErrorResponse for upstream handlers to stop streaming gracefully
            error_obj = getattr(chunk, "error", None)
            try:
                msg = getattr(error_obj, "message", None) or str(error_obj)
            except Exception:
                msg = str(error_obj)
            try:
                code = getattr(error_obj, "code", None)
            except Exception:
                code = None
            try:
                details = getattr(error_obj, "param", None) or getattr(error_obj, "type", None)
            except Exception:
                details = None

            sse_err = SSEErrorResponse(
                data=SSEErrorData(
                    error_code=SSEErrorCode.external_api_error,
                    message=msg or "OpenAI streaming error",
                    details={"code": code, "details": details},
                )
            )
            # Return as a single error chunk for caller to handle immediately
            return [sse_err]

        elif event_type.startswith("response."):
            logger.debug(f"oai_stream_chunk: unhandled event '{event_type}'")

        return processed
