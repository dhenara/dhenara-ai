"""Utilities for converting between Anthropic chat formats and Dhenara message types."""

from __future__ import annotations

import json
from collections.abc import Iterable

from anthropic.types import ContentBlock, Message
from anthropic.types.redacted_thinking_block_param import RedactedThinkingBlockParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.thinking_block_param import ThinkingBlockParam
from anthropic.types.tool_use_block_param import ToolUseBlockParam

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
                    thinking_signature=getattr(content_block, "signature", None),
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
        content_blocks: list[object] = []

        for content in choice.contents:
            if isinstance(content, ChatResponseTextContentItem) and content.text:
                content_blocks.append(TextBlockParam(type="text", text=content.text))
            elif isinstance(content, ChatResponseReasoningContentItem):
                # Anthropic thinking blocks require thinking text + signature
                if content.thinking_text:
                    # Proper thinking block with signature
                    content_blocks.append(
                        ThinkingBlockParam(
                            type="thinking",
                            thinking=content.thinking_text,
                            signature=content.thinking_signature,
                        )
                    )
                elif content.metadata.get("redacted_thinking_data"):
                    # Redacted thinking (when signature but no text)
                    content_blocks.append(
                        RedactedThinkingBlockParam(
                            type="redacted_thinking", data=content.metadata.get("redacted_thinking_data")
                        )
                    )
                elif content.thinking_text:
                    # Fallback: if no signature, emit as text (for cross-provider compatibility)
                    content_blocks.append(
                        TextBlockParam(
                            type="text",
                            text=content.thinking_text,
                        )
                    )
            elif isinstance(content, ChatResponseToolCallContentItem) and content.tool_call:
                tool_call = content.tool_call
                content_blocks.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.name,
                        input=tool_call.arguments,
                    )
                )
            elif isinstance(content, ChatResponseStructuredOutputContentItem):
                output = content.structured_output
                if output.structured_data:
                    content_blocks.append(
                        TextBlockParam(
                            type="text",
                            text=json.dumps(
                                output.structured_data,
                            ),
                        )
                    )

        # Anthropic accepts content as either string or list of block params; keep list form for validation
        if len(content_blocks) == 1 and isinstance(content_blocks[0], TextBlockParam):
            return {"role": "assistant", "content": content_blocks[0].text}

        if content_blocks:
            # SDK accepts a list of block params (they serialize to correct schema)
            return {"role": "assistant", "content": content_blocks}

        return {"role": "assistant", "content": ""}

    @staticmethod
    def choices_to_provider_messages(choices: Iterable[ChatResponseChoice]) -> list[dict[str, object]]:
        return [AnthropicMessageConverter.choice_to_provider_message(choice) for choice in choices]
