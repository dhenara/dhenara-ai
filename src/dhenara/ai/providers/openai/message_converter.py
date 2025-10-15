"""Utilities for converting between OpenAI chat formats and Dhenara message types."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_output_message_param import (
    ResponseOutputMessageParam,
)
from openai.types.responses.response_output_text_param import ResponseOutputTextParam
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
)

from dhenara.ai.providers.openai.constants import OPENAI_USE_RESPONSES_DEFAULT
from dhenara.ai.types.genai import (
    ChatResponseContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseStructuredOutput,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCall,
    ChatResponseToolCallContentItem,
)
from dhenara.ai.types.genai.ai_model import AIModelProviderEnum
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response import ChatResponseChoice

logger = logging.getLogger(__name__)


class OpenAIMessageConverter:
    """Bidirectional converter for OpenAI chat messages."""

    @staticmethod
    def provider_message_to_content_items(
        *,
        message: ChatCompletionMessage,
        role: str,
        index_start: int,
        ai_model_provider: AIModelProviderEnum,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        """Convert an OpenAI provider message into ChatResponseContentItems."""

        if getattr(message, "content", None):
            content_text = message.content
            items: list[ChatResponseContentItem] = []

            # DeepSeek specific reasoning separation (uses <think> tags)
            if ai_model_provider == AIModelProviderEnum.DEEPSEEK and isinstance(content_text, str):
                import re

                think_match = re.search(r"<think>(.*?)</think>", content_text, re.DOTALL)
                if think_match:
                    reasoning_content = think_match.group(1).strip()
                    if reasoning_content:
                        items.append(
                            ChatResponseReasoningContentItem(
                                index=index_start,
                                role=role,
                                thinking_text=reasoning_content,
                            )
                        )
                    answer_content = re.sub(r"<think>.*?</think>", "", content_text, flags=re.DOTALL).strip()
                    if answer_content:
                        content_text = answer_content
                    else:
                        content_text = None

            if structured_output_config is not None and content_text:
                structured_output = ChatResponseStructuredOutput.from_model_output(
                    raw_response=content_text,
                    config=structured_output_config,
                )
                items.append(
                    ChatResponseStructuredOutputContentItem(
                        index=index_start,
                        role=role,
                        structured_output=structured_output,
                    )
                )
            elif content_text:
                items.append(
                    ChatResponseTextContentItem(
                        index=index_start,
                        role=role,
                        text=content_text,
                    )
                )

            return items

        if getattr(message, "tool_calls", None):
            tool_call_items: list[ChatResponseContentItem] = []
            for tool_call in message.tool_calls or []:
                if isinstance(tool_call, dict):
                    tool_payload = tool_call
                else:
                    tool_payload = tool_call.model_dump()

                tool_call_items.append(
                    ChatResponseToolCallContentItem(
                        index=index_start,
                        role=role,
                        tool_call=ChatResponseToolCall.from_openai_format(tool_payload),
                        metadata={},
                    )
                )

            return tool_call_items

        return []

    @staticmethod
    def choice_to_provider_message_chat_api(choice: ChatResponseChoice) -> ChatCompletionAssistantMessageParam:
        """Convert ChatResponseChoice into OpenAI ChatCompletionAssistantMessageParam.

        **DEPRECATED**: OpenAI Chat API path. Use Responses API (`choice_to_provider_message_responses_api`) instead.

        This method converts all content to plain text for legacy Chat API compatibility.
        Reasoning content is flattened to text (no native reasoning support in Chat API).
        Tool calls and structured outputs are preserved when possible.
        """
        logger.warning(
            "OpenAI Chat API converter is deprecated. Use Responses API "
            "(`choice_to_provider_message_responses_api`) for full fidelity."
        )

        content_parts: list[ChatCompletionContentPartTextParam] = []
        tool_calls: list[ChatCompletionMessageToolCallParam] = []

        for item in choice.contents:
            if isinstance(item, ChatResponseTextContentItem) and item.text:
                content_parts.append(ChatCompletionContentPartTextParam(type="text", text=item.text))
            elif isinstance(item, ChatResponseReasoningContentItem):
                # Convert reasoning to text (Chat API doesn't support explicit reasoning)
                text = item.thinking_text or item.thinking_summary
                if text:
                    content_parts.append(ChatCompletionContentPartTextParam(type="text", text=text))
            elif isinstance(item, ChatResponseStructuredOutputContentItem):
                output = item.structured_output
                if output and output.structured_data is not None:
                    content_parts.append(
                        ChatCompletionContentPartTextParam(type="text", text=json.dumps(output.structured_data))
                    )
            elif isinstance(item, ChatResponseToolCallContentItem) and item.tool_call:
                tc = item.tool_call
                tool_calls.append(
                    ChatCompletionMessageToolCallParam(
                        id=tc.id,
                        type="function",
                        function={
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if tc.arguments is not None else "{}",
                        },
                    )
                )

        # When no parts, emit an empty text part to satisfy schema
        if not content_parts and not tool_calls:
            content_parts = [ChatCompletionContentPartTextParam(type="text", text="")]

        return ChatCompletionAssistantMessageParam(
            role="assistant", content=content_parts if content_parts else None, tool_calls=tool_calls or None
        )

    # Responses API output conversion (provider → Dhenara)
    @staticmethod
    def provider_message_to_content_items_responses_api(
        *,
        output_item: object,
        role: str,
        index_start: int,
        ai_model_provider: AIModelProviderEnum,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        """Convert a single Responses API output item into ChatResponseContentItems.

        Handles item types like 'message' (with output_text items), 'reasoning', and 'function_call'.
        For 'message' content, this will also parse structured output when a schema is provided.
        """
        items: list[ChatResponseContentItem] = []

        # Helper to coerce dict-like access from SDK objects
        def _get(obj: object, attr: str, default=None):
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        item_type = _get(output_item, "type", None)

        # Reasoning/thinking blocks
        if item_type == "reasoning":
            # Reasoning/thinking blocks
            thinking_id = _get(output_item, "id", None)
            signature = _get(output_item, "encrypted_content", None)
            status = _get(output_item, "status", None)
            summary_obj = _get(output_item, "summary", None)
            content_obj = _get(output_item, "content", None)
            if isinstance(content_obj, list):
                content_text = " ".join(filter(None, (_get(c, "text", "") for c in content_obj))) or None
            else:
                content_text = _get(content_obj, "text", None)

            if isinstance(summary_obj, list):
                summary = " ".join(filter(None, (_get(s, "text", "") for s in summary_obj))) or None
            else:
                summary = _get(summary_obj, "text", None)

            items.append(
                ChatResponseReasoningContentItem(
                    index=index_start,
                    role=role,
                    thinking_id=thinking_id,
                    thinking_text=content_text,
                    thinking_summary=summary,
                    thinking_signature=signature,
                    thinking_status=status,
                )
            )

        # Function/tool calls
        if item_type in ("function_call", "tool_call", "function.tool_call"):
            call_id = _get(output_item, "id", None) or _get(output_item, "call_id", None)
            # Name can be at top-level or nested under function.name
            name = _get(output_item, "name", None)
            if not name:
                fn_obj = _get(output_item, "function", None)
                if isinstance(fn_obj, dict):
                    name = fn_obj.get("name")
            arguments = _get(output_item, "arguments", None)
            if isinstance(arguments, str):
                try:
                    import json as _json

                    arguments = _json.loads(arguments)
                except Exception:
                    # Keep as raw string if not JSON
                    pass

            items.append(
                ChatResponseToolCallContentItem(
                    index=index_start,
                    role=role,
                    tool_call=ChatResponseToolCall(
                        id=call_id,
                        name=name,
                        arguments=arguments if isinstance(arguments, dict) else {"raw": arguments},
                    ),
                    metadata={},
                )
            )

        # Assistant message with text/structured output content
        if item_type in ("message", "output_message"):
            contents = _get(output_item, "content", None) or []

            # Build a single concatenated text from output_text parts
            text_parts: list[str] = []
            for part in contents or []:
                # Responses SDK objects: part.type == 'output_text', field 'text'
                part_type = _get(part, "type", None)
                if part_type in ("output_text", "text"):
                    part_text = _get(part, "text", None)
                    if part_text:
                        text_parts.append(part_text)

            text_combined = "".join(text_parts).strip() if text_parts else None

            if structured_output_config is not None and text_combined:
                structured_output = ChatResponseStructuredOutput.from_model_output(
                    raw_response=text_combined,
                    config=structured_output_config,
                )
                items.append(
                    ChatResponseStructuredOutputContentItem(
                        index=index_start,
                        role=role,
                        structured_output=structured_output,
                    )
                )
            elif text_combined is not None:
                items.append(
                    ChatResponseTextContentItem(
                        index=index_start,
                        role=role,
                        text=text_combined,
                    )
                )

            return items

        # Fallback: unknown item => try to coerce text if any
        maybe_text = _get(output_item, "text", None)
        if maybe_text:
            items.append(
                ChatResponseTextContentItem(
                    index=index_start,
                    role=role,
                    text=str(maybe_text),
                )
            )
            return items

        return items

    @staticmethod
    def choice_to_provider_message_responses_api(
        choice: ChatResponseChoice,
    ) -> dict[str, object]:
        """Convert ChatResponseChoice into OpenAI Responses API input format.

        Returns a dict containing role='assistant' and 'output' array with proper SDK param types:
        - ResponseOutputMessageParam for text/structured outputs
        - ResponseReasoningItemParam for thinking/reasoning content
        - ResponseFunctionToolCallParam for tool calls

        This maintains full fidelity with Responses API schema, preserving reasoning IDs,
        signatures, summaries, and tool call metadata.
        """
        output_items: list[ResponseOutputMessageParam | ResponseReasoningItemParam | ResponseFunctionToolCallParam] = []

        # Group content by type to build output items
        for item in choice.contents:
            if isinstance(item, ChatResponseTextContentItem) and item.text:
                # ResponseOutputMessage with output_text content
                output_items.append(
                    ResponseOutputMessageParam(
                        id=f"msg_{len(output_items)}",  # Generate ID if not tracked
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[ResponseOutputTextParam(type="output_text", text=item.text, annotations=[])],
                    )
                )
            elif isinstance(item, ChatResponseStructuredOutputContentItem):
                output = item.structured_output
                if output and output.structured_data is not None:
                    # Structured output as output_text JSON
                    output_items.append(
                        ResponseOutputMessageParam(
                            id=f"msg_{len(output_items)}",
                            type="message",
                            role="assistant",
                            status="completed",
                            content=[
                                ResponseOutputTextParam(
                                    type="output_text", text=json.dumps(output.structured_data), annotations=[]
                                )
                            ],
                        )
                    )
            elif isinstance(item, ChatResponseReasoningContentItem):
                # ResponseReasoningItem with thinking content and summary
                thinking_id = item.thinking_id or f"reasoning_{len(output_items)}"
                summary_text = item.thinking_summary or ""
                content_parts = []
                if item.thinking_text:
                    # content is optional per SDK, omit if none
                    content_parts = [
                        ResponseOutputTextParam(type="output_text", text=item.thinking_text, annotations=[])
                    ]
                output_items.append(
                    ResponseReasoningItemParam(
                        id=thinking_id,
                        type="reasoning",
                        summary=[ResponseOutputTextParam(type="output_text", text=summary_text, annotations=[])],
                        content=content_parts or None,
                        encrypted_content=item.thinking_signature,
                        status=item.thinking_status or "completed",
                    )
                )
            elif isinstance(item, ChatResponseToolCallContentItem) and item.tool_call:
                tc = item.tool_call
                output_items.append(
                    ResponseFunctionToolCallParam(
                        id=tc.id or f"call_{len(output_items)}",
                        type="function_call",
                        call_id=tc.id or f"call_{len(output_items)}",
                        name=tc.name,
                        arguments=json.dumps(tc.arguments) if tc.arguments else "{}",
                        status="completed",
                    )
                )

        # If no output items, add an empty message to satisfy schema
        if not output_items:
            output_items.append(
                ResponseOutputMessageParam(
                    id="msg_0",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputTextParam(type="output_text", text="", annotations=[])],
                )
            )

        return {"role": "assistant", "output": output_items}

    @staticmethod
    def choice_to_provider_message_responses_api_input(
        choice: ChatResponseChoice,
    ) -> dict[str, object]:
        """Convert ChatResponseChoice to Responses API INPUT format for multi-turn conversations.

        For prior assistant messages in conversation history, Responses API expects:
        {"role": "assistant", "content": [{"type": "output_text", "text": "..."}]}

        Note: Assistant messages use "output_text" type, not "input_text". This simplifies
        the assistant's prior response to plain text, discarding reasoning IDs and complex
        structure. The full output format with reasoning items is only used in model responses,
        not when feeding history back as input.
        """
        import json as _json

        # Collect all text content (text, reasoning text, structured output as JSON)
        texts: list[str] = []
        for item in choice.contents:
            if isinstance(item, ChatResponseTextContentItem) and item.text:
                texts.append(item.text)
            elif isinstance(item, ChatResponseReasoningContentItem):
                # For reasoning, prefer summary if available, else thinking_text
                if item.thinking_summary:
                    texts.append(f"[Reasoning Summary] {item.thinking_summary}")
                elif item.thinking_text:
                    texts.append(f"[Reasoning] {item.thinking_text}")
            elif isinstance(item, ChatResponseStructuredOutputContentItem) and item.structured_output:
                try:
                    if item.structured_output.structured_data:
                        texts.append(_json.dumps(item.structured_output.structured_data))
                except Exception:
                    pass
            elif isinstance(item, ChatResponseToolCallContentItem):
                # Tool calls need to be represented separately in input format
                # For now, represent as text description
                tc = item.tool_call
                texts.append(f"[Tool Call: {tc.name}({_json.dumps(tc.arguments) if tc.arguments else ''})]")

        # Combine all text into single content
        combined_text = "\n\n".join(texts) if texts else ""

        return {
            "role": "assistant",
            "content": [{"type": "output_text", "text": combined_text}],
        }

    @staticmethod
    def choice_to_provider_message(choice: ChatResponseChoice) -> ChatCompletionAssistantMessageParam:
        if OPENAI_USE_RESPONSES_DEFAULT:
            return OpenAIMessageConverter.choice_to_provider_message_responses_api(choice)
        else:
            return OpenAIMessageConverter.choice_to_provider_message_chat_api(choice)

    @staticmethod
    def choices_to_provider_messages(
        choices: Iterable[ChatResponseChoice],
    ) -> list[ChatCompletionAssistantMessageParam]:
        if OPENAI_USE_RESPONSES_DEFAULT:
            return [OpenAIMessageConverter.choice_to_provider_message_responses_api(choice) for choice in choices]
        else:
            return [OpenAIMessageConverter.choice_to_provider_message_chat_api(choice) for choice in choices]
