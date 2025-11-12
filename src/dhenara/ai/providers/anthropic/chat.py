import json
import logging
from collections.abc import AsyncGenerator, Generator

from anthropic.types import (
    ContentBlock,
    Message,
    MessageStreamEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    SignatureDelta,
    TextDelta,
    ThinkingDelta,
)

from dhenara.ai.providers.anthropic import AnthropicClientBase
from dhenara.ai.providers.anthropic.message_converter import AnthropicMessageConverter
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseContentItem,
    ChatResponseContentItemDelta,
    ChatResponseReasoningContentItem,
    ChatResponseReasoningContentItemDelta,
    ChatResponseTextContentItemDelta,
    ChatResponseToolCall,
    ChatResponseToolCallContentItemDelta,
    ChatResponseUsage,
    StreamingChatResponse,
)
from dhenara.ai.types.shared.api import SSEErrorResponse

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
class AnthropicChat(AnthropicClientBase):
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
            raise ValueError("inputs must be validated with validate_inputs() before api calls")

        messages_list = []
        user = self.config.get_user()

        # Process system instructions
        system_prompt = None
        if instructions:
            if not (isinstance(instructions, dict) and "content" in instructions.keys()):
                raise ValueError(
                    f"Invalid Instructions format. "
                    f"Instructions should be processed and passed in prompt format. Value is {instructions} "
                )
            system_prompt = instructions["content"]  # Extract text from system prompt

        # Add previous messages and current prompt
        if messages is not None:
            # Convert MessageItem objects to Anthropic format
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

        # Prepare API call arguments
        chat_args = {
            "model": self.model_name_in_api_calls,
            "messages": messages_list,
            "stream": self.config.streaming,
        }

        if system_prompt:
            chat_args["system"] = system_prompt

        if user:
            chat_args["metadata"] = {"user_id": user}

        max_output_tokens, max_reasoning_tokens = self.config.get_max_output_tokens(self.model_endpoint.ai_model)
        if max_output_tokens is not None:
            chat_args["max_tokens"] = max_output_tokens

        if max_reasoning_tokens is not None:
            chat_args["thinking"] = {
                "type": "enabled",
                "budget_tokens": max_reasoning_tokens,
            }

        if self.config.options:
            chat_args.update(self.config.options)

        # ---  Tools ---
        if self.config.tools:
            chat_args["tools"] = self.formatter.format_tools(
                tools=self.config.tools,
                model_endpoint=self.model_endpoint,
            )

        if self.config.tool_choice:
            chat_args["tool_choice"] = self.formatter.format_tool_choice(
                tool_choice=self.config.tool_choice,
                model_endpoint=self.model_endpoint,
            )

        # --- Structured Output ---
        # Anthropic uses the tool system for structured output
        if self.config.structured_output:
            # For Anthropic, we need to set up tool calling
            if "tools" not in chat_args:
                chat_args["tools"] = []

            # Add structured output as a tool
            structured_tool = self.formatter.format_structured_output(
                structured_output=self.config.structured_output,
                model_endpoint=self.model_endpoint,
            )
            chat_args["tools"].append(structured_tool)

            # Enforce this tool (when thinking is not enabled; otherwise fall back to auto per API constraint)
            if max_reasoning_tokens is not None:
                # TODO_FUTURE: Revisit this later if API improves in future
                # Currently when enforced tool use in thinking mode,  API flags error as
                # 'Thinking may not be enabled when tool_choice forces tool use.'
                # The irony is that they don't have a structured-output mode either
                chat_args["tool_choice"] = {"type": "auto"}
            else:
                chat_args["tool_choice"] = {
                    "type": "tool",
                    "name": structured_tool["name"],
                }
        else:
            # Take care of tool_choice + thinking mode conflict when no structured output but tools are present
            _tools = chat_args.get("tools", [])
            if max_reasoning_tokens is not None and _tools:
                # TODO_FUTURE: Revisit this later if API improves in future
                # Currently when enforced tool use in thinking mode,  API flags error as
                # 'Thinking may not be enabled when tool_choice forces tool use.'
                # The irony is that they don't have a structured-output mode either
                chat_args["tool_choice"] = {"type": "auto"}

        return {"chat_args": chat_args}

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        chat_args = api_call_params["chat_args"]
        response = self._client.messages.create(**chat_args)
        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        chat_args = api_call_params["chat_args"]
        response = await self._client.messages.create(**chat_args)
        return response

    def do_streaming_api_call_sync(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        chat_args = api_call_params["chat_args"]

        def _wrapper() -> Generator:
            # Use SDK stream context to access final aggregated message
            _args = dict(chat_args)
            _args.pop("stream", None)
            with self._client.messages.stream(**_args) as stream:
                yield from stream
                try:
                    # Get native final message and prefer it for finalization
                    final_msg = stream.get_final_message()
                    self._set_native_final_from_message(final_msg)
                except Exception:
                    # Fallback to legacy reconstruction
                    pass

        return _wrapper()

    async def do_streaming_api_call_async(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        chat_args = api_call_params["chat_args"]

        async def _awrapper() -> AsyncGenerator:
            # Use SDK async stream context to access final aggregated message
            _args = dict(chat_args)
            _args.pop("stream", None)
            async with self._client.messages.stream(**_args) as stream:
                async for event in stream:
                    yield event
                try:
                    final_msg = await stream.get_final_message()
                    self._set_native_final_from_message(final_msg)
                except Exception:
                    pass

        return _awrapper()

    def _set_native_final_from_message(self, final_msg: Message) -> None:
        """Single source of truth: prefer Anthropic SDK final message for artifacts.

        Stores both the provider-native payload (for dai_provider_response.json)
        and the normalized DAI ChatResponse (for dai_response.json via StreamingManager).
        """
        # Store native SDK payload for diagnostics and provenance
        try:
            self.streaming_manager.native_final_response_sdk = (
                final_msg.model_dump() if hasattr(final_msg, "model_dump") else None
            )
        except Exception:
            self.streaming_manager.native_final_response_sdk = None

        # Parse to our canonical ChatResponse (includes provider_response for artifacts writer)
        self.streaming_manager.native_final_response_dai = self.parse_response(final_msg)

    def parse_stream_chunk(
        self,
        chunk: MessageStreamEvent,
    ) -> StreamingChatResponse | SSEErrorResponse | None:
        """Handle streaming response with progress tracking and final response"""

        processed_chunks = []

        # self.streaming_manager.message_metadata  is used to preserve params of initial message across chunks
        if isinstance(chunk, RawMessageStartEvent):
            message = chunk.message

            # Initialize message metadata
            self.streaming_manager.message_metadata = {
                "id": message.id,
                "model": message.model,
                "role": message.role,
                "type": type,
                "index": 0,  # Only one choice from Antropic
            }

            # Anthropic has a wieded way of reporint usage on streaming
            # On message_start, usage will have input tokens and few output tokens
            _usage = chunk.message.usage
            if _usage:
                # Initialize usage in self.streaming_manager
                usage = ChatResponseUsage(
                    total_tokens=0,
                    prompt_tokens=_usage.input_tokens,
                    completion_tokens=_usage.output_tokens,
                )
                self.streaming_manager.update_usage(usage)

            # Track active tool_use block indices for correct delta routing
            if not hasattr(self.streaming_manager, "anthropic_tool_use_indices"):
                self.streaming_manager.anthropic_tool_use_indices = set()

        elif isinstance(chunk, RawContentBlockStartEvent):
            block_type = chunk.content_block.type
            if block_type == "redacted_thinking":
                content_deltas = [
                    ChatResponseReasoningContentItem(
                        index=chunk.index,
                        role=self.streaming_manager.message_metadata["role"],
                        metadata={
                            "redacted_thinking_data": chunk.content_block.data,
                        },
                    )
                ]

                choice_deltas = [
                    ChatResponseChoiceDelta(
                        index=self.streaming_manager.message_metadata["index"],
                        content_deltas=content_deltas,
                        metadata={},
                    )
                ]

                response_chunk = self.streaming_manager.update(choice_deltas=choice_deltas)
                stream_response = StreamingChatResponse(
                    id=self.streaming_manager.message_metadata["id"],
                    data=response_chunk,
                )

                processed_chunks.append(stream_response)
            elif block_type in ["text", "thinking"]:
                pass
            elif block_type == "tool_use":
                # Initialize a tool call item with id and name; args will stream via deltas
                tool_id = getattr(chunk.content_block, "id", None)
                tool_name = getattr(chunk.content_block, "name", None)
                # Remember this index as a tool_use block for subsequent deltas
                try:
                    self.streaming_manager.anthropic_tool_use_indices.add(chunk.index)
                except Exception:
                    pass

                content_deltas = [
                    ChatResponseToolCallContentItemDelta(
                        index=chunk.index,
                        role=self.streaming_manager.message_metadata["role"],
                        tool_call=ChatResponseToolCall(
                            call_id=tool_id,
                            id=None,
                            name=tool_name,
                            arguments={},
                        ),
                        metadata={"tool_use_start": True},
                    )
                ]

                choice_deltas = [
                    ChatResponseChoiceDelta(
                        index=self.streaming_manager.message_metadata["index"],
                        content_deltas=content_deltas,
                        metadata={},
                    )
                ]
                response_chunk = self.streaming_manager.update(choice_deltas=choice_deltas)
                stream_response = StreamingChatResponse(
                    id=self.streaming_manager.message_metadata["id"],
                    data=response_chunk,
                )
                processed_chunks.append(stream_response)
            else:
                logger.debug(f"anthropic: Unhandled content_block_type {block_type}")

        elif isinstance(chunk, RawContentBlockDeltaEvent):
            # Tool use arguments typically stream as input_json deltas; detect by tracked indices
            is_tool_use = False
            try:
                is_tool_use = chunk.index in getattr(self.streaming_manager, "anthropic_tool_use_indices", set())
            except Exception:
                is_tool_use = False
            if is_tool_use:
                # Attempt to read partial JSON from delta; if dict, serialize chunk
                delta = chunk.delta
                partial = getattr(delta, "partial_json", None)
                if partial is None:
                    # Fallbacks for different SDK shapes
                    partial = getattr(delta, "delta", None)
                try:
                    if partial is None:
                        arguments_delta = ""
                    elif isinstance(partial, str):
                        arguments_delta = partial
                    else:
                        arguments_delta = json.dumps(partial)
                except Exception:
                    arguments_delta = str(partial)

                content_deltas = [
                    ChatResponseToolCallContentItemDelta(
                        index=chunk.index,
                        role=self.streaming_manager.message_metadata["role"],
                        arguments_delta=arguments_delta,
                        metadata={"tool_use_args_delta": True},
                    )
                ]
            else:
                content_deltas = [
                    self.process_content_item_delta(
                        index=chunk.index,
                        role=self.streaming_manager.message_metadata["role"],
                        delta=chunk.delta,
                    )
                ]

            choice_deltas = [
                ChatResponseChoiceDelta(
                    index=self.streaming_manager.message_metadata["index"],
                    content_deltas=content_deltas,
                    metadata={},
                )
            ]
            response_chunk = self.streaming_manager.update(choice_deltas=choice_deltas)
            stream_response = StreamingChatResponse(
                id=self.streaming_manager.message_metadata["id"],
                data=response_chunk,
            )
            processed_chunks.append(stream_response)
        elif isinstance(chunk, RawContentBlockStopEvent):
            # For tool_use, emit a finalize delta to parse accumulated arguments buffer
            finalize_tool = False
            try:
                finalize_tool = chunk.index in getattr(self.streaming_manager, "anthropic_tool_use_indices", set())
            except Exception:
                finalize_tool = False
            if finalize_tool:
                content_deltas = [
                    ChatResponseToolCallContentItemDelta(
                        index=chunk.index,
                        role=self.streaming_manager.message_metadata["role"],
                        metadata={"finalize_tool_call": True},
                    )
                ]
                choice_deltas = [
                    ChatResponseChoiceDelta(
                        index=self.streaming_manager.message_metadata["index"],
                        content_deltas=content_deltas,
                        metadata={},
                    )
                ]
                response_chunk = self.streaming_manager.update(choice_deltas=choice_deltas)
                stream_response = StreamingChatResponse(
                    id=self.streaming_manager.message_metadata["id"],
                    data=response_chunk,
                )
                processed_chunks.append(stream_response)
                # Clear the index tracking for this block
                try:
                    self.streaming_manager.anthropic_tool_use_indices.discard(chunk.index)
                except Exception:
                    pass
        elif isinstance(chunk, RawMessageDeltaEvent):
            # Update output tokens
            self.streaming_manager.usage.completion_tokens += chunk.usage.output_tokens
            self.streaming_manager.usage.total_tokens = (
                self.streaming_manager.usage.prompt_tokens + self.streaming_manager.usage.completion_tokens
            )

            # Update choice metatdata
            choice_deltas = [
                ChatResponseChoiceDelta(
                    index=self.streaming_manager.message_metadata["index"],
                    finish_reason=chunk.delta.stop_reason,
                    stop_sequence=chunk.delta.stop_sequence,
                    content_deltas=[],
                    metadata={},
                )
            ]
            response_chunk = self.streaming_manager.update(choice_deltas=choice_deltas)
            stream_response = StreamingChatResponse(
                id=self.streaming_manager.message_metadata["id"],
                data=response_chunk,
            )
            processed_chunks.append(stream_response)

        elif isinstance(chunk, RawMessageStopEvent):
            pass
        else:
            logger.debug(f"anthropic: Unhandled message type {chunk.type}")

        return processed_chunks

    # API has stopped streaming, get final response

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
            provider_response=self.serialize_provider_response(response),
            choices=[
                ChatResponseChoice(
                    index=0,  # Only one choice
                    finish_reason=response.stop_reason,
                    stop_sequence=response.stop_sequence,
                    contents=[
                        self.process_content_item(
                            index=content_index,
                            role=response.role,
                            content_item=content_item,
                        )
                        for content_index, content_item in enumerate(response.content)
                    ],
                    metadata={},  # Choice metadata
                ),
            ],
            # Response Metadata
            metadata=AIModelCallResponseMetaData(
                streaming=False,
                duration_seconds=0,  # TODO
                provider_metadata={
                    "id": response.id,
                },
            ),
        )

    def process_content_item(
        self,
        index: int,
        role: str,
        content_item: ContentBlock,
    ) -> ChatResponseContentItem:
        converted_items = AnthropicMessageConverter._content_block_to_items(
            content_block=content_item,
            index=index,
            role=role,
            structured_output_config=self.config.structured_output,
        )

        if converted_items:
            # Anthropic content blocks typically map to a single item; return the first.
            return converted_items[0]

        return self.get_unknown_content_type_item(
            index=index,
            role=role,
            unknown_item=content_item,
            streaming=False,
        )

    def process_content_item_delta(
        self,
        index: int,
        role: str,
        delta,
    ) -> ChatResponseContentItemDelta:
        if isinstance(delta, TextDelta):
            return ChatResponseTextContentItemDelta(
                index=index,
                role=role,
                text_delta=delta.text,
            )
        elif isinstance(delta, ThinkingDelta):
            return ChatResponseReasoningContentItemDelta(
                index=index,
                role=role,
                text_delta=delta.thinking,
                thinking_signature=None,
                metadata={},
            )
        elif isinstance(delta, SignatureDelta):
            return ChatResponseReasoningContentItemDelta(
                index=index,
                role=role,
                text_delta="",
                thinking_signature=delta.signature,
                metadata={},
            )
        # TODO: Tools Not supported in streaming yet
        else:
            return self.get_unknown_content_type_item(
                index=index,
                role=role,
                unknown_item=delta,
                streaming=True,
            )
