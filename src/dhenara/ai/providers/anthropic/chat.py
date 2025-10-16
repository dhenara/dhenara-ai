import logging

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

            # Enforce this tool
            if max_reasoning_tokens is not None:
                # TODO_FUTURE: Revisit this later if API improves in future
                # Currently when enforced tool use in thiking mode,  API flags error as
                # 'Thinking may not be enabled when tool_choice forces tool use.'
                # The irony is that they don't have a structured-output mode either
                chat_args["tool_choice"] = {"type": "auto"}
            else:
                chat_args["tool_choice"] = {
                    "type": "tool",
                    "name": structured_tool["name"],
                }

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
        stream = self._client.messages.create(**chat_args)
        return stream

    async def do_streaming_api_call_async(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        chat_args = api_call_params["chat_args"]
        stream = await self._client.messages.create(**chat_args)
        return stream

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
            else:
                logger.debug(f"anthropic: Unhandled content_block_type {block_type}")

        elif isinstance(chunk, RawContentBlockDeltaEvent):
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
                thinking_text_delta=delta.thinking,
                thinking_signature=None,  # TODO: Take care of signature
                metadata={},
            )
        elif isinstance(delta, SignatureDelta):
            return ChatResponseReasoningContentItemDelta(
                index=index,
                role=role,
                thinking_text_delta="",
                metadata={
                    "signature": delta.signature,
                },
            )
        # TODO: Tools Not supported in streaming yet
        else:
            return self.get_unknown_content_type_item(
                index=index,
                role=role,
                unknown_item=delta,
                streaming=True,
            )
