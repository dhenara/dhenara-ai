"""Utilities for converting between Anthropic chat formats and Dhenara message types."""

from __future__ import annotations

import json
import logging

from anthropic.types import ContentBlock, Message
from anthropic.types.redacted_thinking_block_param import RedactedThinkingBlockParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.thinking_block_param import ThinkingBlockParam
from anthropic.types.tool_use_block_param import ToolUseBlockParam

from dhenara.ai.providers.base import BaseMessageConverter
from dhenara.ai.types.genai import (
    ChatMessageContentPart,
    ChatResponseContentItem,
    ChatResponseContentItemType,
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

logger = logging.getLogger(__name__)


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

            # UPDATE: Anthropic does NOT return structured output via plain text. Only tool_use blocks carry
            # structured payloads when we define a tool for structured output.
            # Therefore, always treat text as plain text regardless of structured_output_config.
            #
            # OLD CODE --- IGNORE ---
            # if structured_output_config is not None:
            #     # Parse structured output from plain text and retain original part for round-trip
            #     parsed_data, error, post_processed = ChatResponseStructuredOutput._parse_and_validate(
            #         text_value,
            #         structured_output_config,
            #     )
            #     structured_output = ChatResponseStructuredOutput(
            #         config=structured_output_config,
            #         structured_data=parsed_data,
            #         raw_data=text_value,
            #         parse_error=error,
            #         post_processed=post_processed,
            #     )
            #     return [
            #         ChatResponseStructuredOutputContentItem(
            #             index=index,
            #             role=role,
            #             structured_output=structured_output,
            #             message_contents=[
            #                 ChatMessageContentPart(
            #                     type="text",
            #                     text=text_value,
            #                     annotations=None,
            #                     metadata=None,
            #                 )
            #             ],
            #         )
            #     ]

            return [
                ChatResponseTextContentItem(
                    index=index,
                    role=role,
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
            # Preserve thinking via message_contents with a thinking part; keep signature/id
            thinking_text = getattr(content_block, "thinking", "")
            return [
                ChatResponseReasoningContentItem(
                    index=index,
                    role=role,
                    thinking_signature=getattr(content_block, "signature", None),
                    thinking_id=getattr(content_block, "id", None),
                    message_contents=[
                        ChatMessageContentPart(type="thinking", text=thinking_text, annotations=None, metadata=None)
                    ],
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
            # IMPORTANT: ChatResponseReasoningContentItem subclasses ChatResponseTextContentItem.
            # We must handle reasoning BEFORE generic text; otherwise reasoning items will be treated as plain text
            # and their thinking blocks/signatures won't round-trip.
            if content.type == ChatResponseContentItemType.REASONING:
                # Anthropic thinking blocks require thinking text + signature
                # Prefer message_contents parts of type 'thinking' when present.
                thinking_text = None
                if content.message_contents:
                    parts = content.message_contents
                    texts = [p.text for p in parts if p.text]
                    if texts:
                        thinking_text = "".join(texts)

                if thinking_text and content.thinking_signature:
                    # Proper thinking block with signature
                    content_blocks.append(
                        ThinkingBlockParam(
                            type="thinking",
                            thinking=thinking_text,
                            signature=content.thinking_signature if same_provider else None,
                        )
                    )
                elif content.thinking_summary:
                    # Redacted thinking (when signature but no text)
                    # If represented as summary parts, try to map redacted_thinking type to redacted block
                    if isinstance(content.thinking_summary, list):
                        rt = next(
                            (p for p in content.thinking_summary if p.type == "redacted_thinking"),
                            None,
                        )
                        if rt is not None:
                            content_blocks.append(
                                RedactedThinkingBlockParam(
                                    type="redacted_thinking",
                                    data=rt.metadata,
                                )
                            )
                            # handled as redacted
                            continue

                    # If no redacted part but a textual summary exists, fallback to text emission
                    summary_text = None
                    if isinstance(content.thinking_summary, list):
                        summary_texts = [p.text for p in content.thinking_summary if p.text]
                        summary_text = "".join(summary_texts) if summary_texts else None

                    if summary_text:
                        if same_provider and not content.thinking_signature:
                            raise ValueError(
                                "Anthropic: missing thinking signature for reasoning content in strict mode.",
                            )
                        content_blocks.append(TextBlockParam(type="text", text=summary_text))
                    elif thinking_text:
                        # No signature present: only allowed for cross-provider replay
                        if same_provider:
                            raise ValueError(
                                "Anthropic: missing thinking signature for reasoning content in strict mode.",
                            )
                        content_blocks.append(TextBlockParam(type="text", text=thinking_text))
                else:
                    # No text, no summary; skip silently (encrypted-only with missing parts)
                    logger.warning("ant_convert: reasoning item had neither text nor summary")

            elif content.type == ChatResponseContentItemType.TEXT:
                # Replay message_contents if available for better round-tripping
                if content.message_contents:
                    # Accept both legacy 'text' and unified 'output_text' part types
                    text_parts = [
                        part.text
                        for part in content.message_contents
                        # if getattr(part, "text", None) and getattr(part, "type", None) in ("text", "output_text")
                    ]
                    content_blocks.extend([TextBlockParam(type="text", text=tp) for tp in text_parts if tp])

                else:
                    # content_blocks.append(TextBlockParam(type="text", text=content.text))
                    logger.warning("ant_convert: TextContentItem has no message_contents")

            elif content.type == ChatResponseContentItemType.TOOL_CALL:
                tool_call = content.tool_call
                content_blocks.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        id=tool_call.call_id,
                        name=tool_call.name,
                        input=tool_call.arguments,
                    )
                )

            elif content.type == ChatResponseContentItemType.STRUCTURED_OUTPUT:
                if same_provider:
                    # INFO:
                    # Anthropic structured output is generated via tool use, so its message_contents will be be empty.
                    # Thus structured must be paut back as text by removing 'id' for round-trip.
                    # NOTE:
                    # Do NOT try to convert it as ToolBlock, else API will fail expecting a `result` for the tool_call
                    # Live with this dirty workaround until Anthropic natively supports structured output blocks in API

                    raw_data = content.structured_output.raw_data
                    if raw_data:
                        try:
                            # # Make a copy to avoid mutating the original
                            # raw_data_copy = dict(raw_data)
                            # raw_data_copy.pop("id", None)  # Remove 'id` for more native text block
                            # content_blocks.append(
                            #     TextBlockParam(
                            #         type="text",
                            #         text=json.dumps(raw_data_copy),
                            #     )
                            # )

                            # Just return the fn args to look like a pure text block with only structured args
                            fn_args = raw_data.get("input")
                            content_blocks.append(
                                TextBlockParam(
                                    type="text",
                                    text=json.dumps(fn_args),
                                )
                            )
                            continue  # Proceed to next content item
                        except Exception as e:
                            logger.warning(f"ant_convert: Failed to serialize structured output raw_data to text: {e}")

                    # Fallback: serialize structured_data as JSON text
                    output = content.structured_output
                    if output and output.structured_data:
                        content_blocks.append(
                            TextBlockParam(
                                type="text",
                                text=json.dumps(output.structured_data),
                            )
                        )

                else:
                    # Prefer replaying message_contents for round-trip fidelity
                    if content.message_contents:
                        content_blocks.extend(
                            [
                                TextBlockParam(type="text", text=part.text)
                                for part in content.message_contents
                                if part.type == "text" and part.text
                            ]
                        )
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

            else:
                logger.error(
                    f"ant_convert: Unhandled content item type: {content.type} at index {content.index}",
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
