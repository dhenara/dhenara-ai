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


class OpenAIMessageConverterCHATAPI:
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
                                message_contents=[
                                    {
                                        "type": "thinking",
                                        "text": reasoning_content,
                                    }
                                ],
                            )
                        )
                    answer_content = re.sub(r"<think>.*?</think>", "", content_text, flags=re.DOTALL).strip()
                    if answer_content:
                        content_text = answer_content
                    else:
                        content_text = None

            if structured_output_config is not None and content_text:
                parsed_data, error, post_processed = ChatResponseStructuredOutput._parse_and_validate(
                    content_text, structured_output_config
                )

                structured_output = ChatResponseStructuredOutput(
                    config=structured_output_config,
                    structured_data=parsed_data,
                    raw_data=content_text,  # Keep original response regardless of parsing
                    parse_error=error,
                    post_processed=post_processed,
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
                        message_contents=[{"type": "text", "text": content_text}],
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

                _args = tool_payload.get("function", {}).get("arguments")
                _parsed_args = ChatResponseToolCall.parse_args_str_or_dict(_args)

                tool_call = ChatResponseToolCall(
                    id=tool_payload.get("id"),
                    name=tool_payload.get("function", {}).get("name"),
                    arguments=_parsed_args.get("arguments_dict"),
                    raw_data=_parsed_args.get("raw_data"),
                    parse_error=_parsed_args.get("parse_error"),
                )

                tool_call_items.append(
                    ChatResponseToolCallContentItem(
                        index=index_start,
                        role=role,
                        tool_call=tool_call,
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
                t = content.get_text() if hasattr(content, "get_text") else None
                if t:
                    text_parts.append(t)
            elif isinstance(content, ChatResponseReasoningContentItem):
                t = content.get_text() if hasattr(content, "get_text") else None
                if t:
                    reasoning_parts.append(t)
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
        return [OpenAIMessageConverterCHATAPI.choice_to_provider_message(choice) for choice in choices]
