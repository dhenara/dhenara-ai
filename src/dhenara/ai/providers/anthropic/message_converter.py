"""Utilities for converting between Anthropic chat formats and Dhenara message types."""

from __future__ import annotations

import json
from collections.abc import Iterable

from anthropic.types import ContentBlock, Message

from dhenara.ai.types.genai import (
    ChatResponseContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseStructuredOutput,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCallContentItem,
)
from dhenara.ai.types.genai.dhenara import ChatResponseToolCall
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response import ChatResponseChoice


class AnthropicMessageConverter:
    """Bidirectional converter for Anthropic messages."""

    @staticmethod
    def provider_message_to_content_items(
        *,
        message: Message,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        items: list[ChatResponseContentItem] = []
        for index, content in enumerate(message.content):
            items.extend(
                AnthropicMessageConverter.content_block_to_items(
                    content_block=content,
                    index=index,
                    role=message.role,
                    structured_output_config=structured_output_config,
                )
            )

        return items

    @staticmethod
    def content_block_to_items(
        *,
        content_block: ContentBlock,
        index: int,
        role: str,
        structured_output_config: StructuredOutputConfig | None,
    ) -> list[ChatResponseContentItem]:
        if content_block.type == "text":
            return [
                ChatResponseTextContentItem(
                    index=index,
                    role=role,
                    text=getattr(content_block, "text", ""),
                )
            ]

        if content_block.type == "thinking":
            return [
                ChatResponseReasoningContentItem(
                    index=index,
                    role=role,
                    thinking_text=getattr(content_block, "thinking", ""),
                    metadata={"signature": getattr(content_block, "signature", None)},
                )
            ]

        if content_block.type == "redacted_thinking":
            return [
                ChatResponseReasoningContentItem(
                    index=index,
                    role=role,
                    metadata={"redacted_thinking_data": getattr(content_block, "data", None)},
                )
            ]

        if content_block.type == "tool_use":
            raw_response = content_block.model_dump()

            try:
                tool_call = ChatResponseToolCall.from_anthropic_format(raw_response)
            except Exception:
                tool_call = None

            if structured_output_config is not None:
                structured_output = ChatResponseStructuredOutput.from_tool_call(
                    raw_response=raw_response,
                    tool_call=tool_call,
                    config=structured_output_config,
                )

                return [
                    ChatResponseStructuredOutputContentItem(
                        index=index,
                        role=role,
                        structured_output=structured_output,
                    )
                ]

            return [
                ChatResponseToolCallContentItem(
                    index=index,
                    role=role,
                    tool_call=tool_call,
                    metadata={},
                )
            ]

        return []

    @staticmethod
    def choice_to_provider_message(choice: ChatResponseChoice) -> dict[str, object]:
        content_blocks: list[dict[str, object]] = []

        for content in choice.contents:
            if isinstance(content, ChatResponseTextContentItem) and content.text:
                content_blocks.append({"type": "text", "text": content.text})
            elif isinstance(content, ChatResponseReasoningContentItem):
                if content.thinking_text:
                    content_blocks.append({"type": "text", "text": content.thinking_text})
                elif content.metadata.get("redacted_thinking_data"):
                    content_blocks.append(
                        {
                            "type": "redacted_thinking",
                            "data": content.metadata.get("redacted_thinking_data"),
                        }
                    )
            elif isinstance(content, ChatResponseToolCallContentItem) and content.tool_call:
                tool_call = content.tool_call
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments,
                    }
                )
            elif isinstance(content, ChatResponseStructuredOutputContentItem):
                output = content.structured_output
                if output.structured_data:
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": json.dumps(output.structured_data),
                        }
                    )

        if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
            return {"role": "assistant", "content": content_blocks[0]["text"]}

        if content_blocks:
            return {"role": "assistant", "content": content_blocks}

        return {"role": "assistant", "content": ""}

    @staticmethod
    def choices_to_provider_messages(choices: Iterable[ChatResponseChoice]) -> list[dict[str, object]]:
        return [AnthropicMessageConverter.choice_to_provider_message(choice) for choice in choices]
