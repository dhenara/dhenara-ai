"""Utilities for converting between Anthropic chat formats and Dhenara message types."""

from __future__ import annotations

import json

from anthropic.types import ContentBlock, Message
from anthropic.types.redacted_thinking_block_param import RedactedThinkingBlockParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.thinking_block_param import ThinkingBlockParam
from anthropic.types.tool_use_block_param import ToolUseBlockParam

from dhenara.ai.providers.base import BaseMessageConverter
from dhenara.ai.types.genai import (
    ChatMessageContentPart,
    ChatResponseContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseStructuredOutput,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCallContentItem,
)
from dhenara.ai.types.genai.ai_model import AIModelEndpoint, AIModelProviderEnum
from dhenara.ai.types.genai.dhenara import ChatResponseToolCall
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response import ChatResponse, ChatResponseChoice


class AnthropicMessageConverter(BaseMessageConverter):
    """Bidirectional converter for Anthropic messages."""

    @staticmethod
    def provider_message_to_dai_content_items(
        *,
        message: Message,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        items: list[ChatResponseContentItem] = []
        for index, content in enumerate(message.content):
            items.extend(
                AnthropicMessageConverter._content_block_to_items(
                    content_block=content,
                    index=index,
                    role=message.role,
                    structured_output_config=structured_output_config,
                )
            )

        return items

    @staticmethod
    def _content_block_to_items(
        *,
        content_block: ContentBlock,
        index: int,
        role: str,
        structured_output_config: StructuredOutputConfig | None,
    ) -> list[ChatResponseContentItem]:
        if content_block.type == "text":
            text_value = getattr(content_block, "text", "")
            if structured_output_config is not None:
                # Parse structured output from plain text and retain original part for round-trip
                parsed_data, error, post_processed = ChatResponseStructuredOutput._parse_and_validate(
                    text_value,
                    structured_output_config,
                )
                structured_output = ChatResponseStructuredOutput(
                    config=structured_output_config,
                    structured_data=parsed_data,
                    raw_data=text_value,
                    parse_error=error,
                    post_processed=post_processed,
                )
                return [
                    ChatResponseStructuredOutputContentItem(
                        index=index,
                        role=role,
                        structured_output=structured_output,
                        message_contents=[
                            ChatMessageContentPart(
                                type="text",
                                text=text_value,
                                annotations=None,
                                metadata=None,
                            )
                        ],
                    )
                ]
            return [
                ChatResponseTextContentItem(
                    index=index,
                    role=role,
                    text=text_value,
                    message_contents=[
                        ChatMessageContentPart(
                            type="text",
                            text=text_value,
                            annotations=None,
                            metadata=None,
                        )
                    ],
                )
            ]

        if content_block.type == "thinking":
            # Preserve thinking_text; Anthropic has signature and id
            return [
                ChatResponseReasoningContentItem(
                    index=index,
                    role=role,
                    thinking_text=getattr(content_block, "thinking", ""),
                    thinking_signature=getattr(content_block, "signature", None),
                    thinking_id=getattr(content_block, "id", None),
                )
            ]

        if content_block.type == "redacted_thinking":
            # Represent redacted thinking as a summary part with type 'redacted_thinking'
            redacted = getattr(content_block, "data", None)
            return [
                ChatResponseReasoningContentItem(
                    index=index,
                    role=role,
                    thinking_summary=[
                        ChatMessageContentPart(type="redacted_thinking", text=None, annotations=None, metadata=redacted)
                    ],
                )
            ]

        if content_block.type == "tool_use":
            raw_response = content_block.model_dump()

            try:
                _args = raw_response.get("input")
                _parsed_args = ChatResponseToolCall.parse_args_str_or_dict(_args)

                tool_call = ChatResponseToolCall(
                    call_id=raw_response.get("id"),
                    id=None,
                    name=raw_response.get("name"),
                    arguments=_parsed_args.get("arguments_dict"),
                    raw_data=_parsed_args.get("raw_data"),
                    parse_error=_parsed_args.get("parse_error"),
                )
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
    def dai_choice_to_provider_message(
        choice: ChatResponseChoice,
        model_endpoint: AIModelEndpoint,
        source_provider: AIModelProviderEnum,
    ) -> dict[str, object]:
        content_blocks: list[object] = []

        same_provider = True if str(source_provider) == str(model_endpoint.ai_model.provider) else False

        for content in choice.contents:
            if isinstance(content, ChatResponseTextContentItem):
                # Replay message_contents if available for better round-tripping
                if content.message_contents:
                    for part in content.message_contents:
                        if part.type == "text" and part.text:
                            content_blocks.append(TextBlockParam(type="text", text=part.text))
                elif content.text:
                    content_blocks.append(TextBlockParam(type="text", text=content.text))
            elif isinstance(content, ChatResponseReasoningContentItem):
                # Anthropic thinking blocks require thinking text + signature
                if content.thinking_text and content.thinking_signature:
                    # Proper thinking block with signature
                    content_blocks.append(
                        ThinkingBlockParam(
                            type="thinking",
                            thinking=content.thinking_text,
                            signature=content.thinking_signature if same_provider else None,
                        )
                    )
                elif content.thinking_summary:
                    # Redacted thinking (when signature but no text)
                    # If represented as summary parts, try to map redacted_thinking type to redacted block
                    rt = next(
                        (p for p in content.thinking_summary if getattr(p, "type", "") == "redacted_thinking"),
                        None,
                    )
                    if rt is not None:
                        content_blocks.append(
                            RedactedThinkingBlockParam(type="redacted_thinking", data=getattr(rt, "metadata", None))
                        )
                    elif content.thinking_text:
                        if same_provider:
                            raise ValueError(
                                "Anthropic: missing thinking signature for reasoning content in strict mode."
                            )
                        content_blocks.append(TextBlockParam(type="text", text=content.thinking_text))
                elif content.thinking_text:
                    if same_provider:
                        raise ValueError("Anthropic: missing thinking signature for reasoning content in strict mode.")
                    # Fallback: if no signature, emit as text (cross-provider compatibility)
                    content_blocks.append(TextBlockParam(type="text", text=content.thinking_text))
            elif isinstance(content, ChatResponseToolCallContentItem) and content.tool_call:
                tool_call = content.tool_call
                content_blocks.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        id=tool_call.call_id,
                        name=tool_call.name,
                        input=tool_call.arguments,
                    )
                )
            elif isinstance(content, ChatResponseStructuredOutputContentItem):
                # Prefer replaying message_contents for round-trip fidelity
                if content.message_contents:
                    for part in content.message_contents:
                        if part.type == "text" and part.text:
                            content_blocks.append(TextBlockParam(type="text", text=part.text))
                else:
                    # Fallback: serialize structured_data as JSON text
                    output = content.structured_output
                    if output and output.structured_data:
                        content_blocks.append(
                            TextBlockParam(
                                type="text",
                                text=json.dumps(output.structured_data),
                            )
                        )

        if content_blocks:
            # SDK accepts a list of block params (they serialize to correct schema)
            return {"role": "assistant", "content": content_blocks}

        return {"role": "assistant", "content": ""}

    @staticmethod
    def dai_response_to_provider_message(
        dai_response: ChatResponse,
        model_endpoint: object | None = None,
    ) -> dict[str, object] | list[dict[str, object]]:
        """Convert a complete ChatResponse to Anthropic provider message format.

        Uses the first choice as the assistant message content, preserving
        reasoning blocks (thinking/redacted_thinking), tool_use, and text.
        """
        if not dai_response or not dai_response.choices:
            return {"role": "assistant", "content": ""}
        return AnthropicMessageConverter.dai_choice_to_provider_message(
            choice=dai_response.choices[0],
            model_endpoint=model_endpoint,
            source_provider=dai_response.provider,
        )
