"""Utilities for converting between OpenAI chat formats and Dhenara message types."""

from __future__ import annotations

import json
from collections.abc import Iterable

from openai.types.chat import ChatCompletionMessage

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
    def choice_to_provider_message(choice: ChatResponseChoice) -> dict[str, object]:
        """Convert ChatResponseChoice into OpenAI-compatible assistant message."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_payload: list[dict[str, object]] = []

        for content in choice.contents:
            if isinstance(content, ChatResponseTextContentItem):
                if content.text:
                    text_parts.append(content.text)
            elif isinstance(content, ChatResponseReasoningContentItem):
                # TODO: Take care of proper conversion back to openai format
                if content.thinking_text:
                    reasoning_parts.append(content.thinking_text)
            elif isinstance(content, ChatResponseToolCallContentItem):
                tool_call = content.tool_call
                tool_calls_payload.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                )
            elif isinstance(content, ChatResponseStructuredOutputContentItem):
                output = content.structured_output
                if output.structured_data:
                    text_parts.append(json.dumps(output.structured_data))

        message: dict[str, object] = {"role": "assistant"}

        if text_parts:
            message["content"] = "\n".join(text_parts)
        elif reasoning_parts:
            message["content"] = "\n".join(reasoning_parts)
        else:
            message["content"] = None

        if tool_calls_payload:
            message["tool_calls"] = tool_calls_payload

        return message

    @staticmethod
    def choices_to_provider_messages(choices: Iterable[ChatResponseChoice]) -> list[dict[str, object]]:
        return [OpenAIMessageConverter.choice_to_provider_message(choice) for choice in choices]

    # Responses API output conversion
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
