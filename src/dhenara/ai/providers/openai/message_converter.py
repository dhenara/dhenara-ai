"""Utilities for converting between OpenAI chat formats and Dhenara message types."""

from __future__ import annotations

import logging
from typing import Any

from openai.types.responses import (
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)

from dhenara.ai.providers.base import BaseMessageConverter
from dhenara.ai.types.genai import (
    ChatMessageContentPart,
    ChatResponseContentItem,
    ChatResponseContentItemType,
    ChatResponseGenericContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseStructuredOutput,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCall,
    ChatResponseToolCallContentItem,
)
from dhenara.ai.types.genai.ai_model import AIModelEndpoint, AIModelProviderEnum
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response import ChatResponse, ChatResponseChoice

logger = logging.getLogger(__name__)


class OpenAIMessageConverter(BaseMessageConverter):
    """Bidirectional converter for OpenAI chat messages."""

    @staticmethod
    def provider_message_to_dai_content_items(
        *,
        message: Any,
        role: str,
        index_start: int,
        ai_model_provider: AIModelProviderEnum,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        content_index = index_start
        content_items: list[ChatResponseContentItem] = []
        for item in message:  # message is `response.output` list
            converted = OpenAIMessageConverter.provider_message_item_to_dai_content_item(
                message_item=item,
                role="assistant",
                index=content_index,
                ai_model_provider=ai_model_provider,
                structured_output_config=structured_output_config,
            )
            if not converted:
                continue
            content_items.append(converted)
            content_index += 1

        return content_items

    @staticmethod
    def provider_message_item_to_dai_content_item(
        *,
        message_item: Any,
        role: str,
        index: int,
        ai_model_provider: AIModelProviderEnum,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> ChatResponseContentItem:
        """Convert a single Responses API output item into ChatResponseContentItems.

        Handles item types like 'message' (with output_text items), 'reasoning', and 'function_call'.
        For 'message' content, this will also parse structured output when a schema is provided.
        """
        output_item = message_item

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

            # Convert OpenAI reasoning summary to list[ChatMessageContentPart] | None
            if isinstance(summary_obj, list):
                summary_list: list[ChatMessageContentPart] = []
                for s in summary_obj:
                    s_dict = s.model_dump() if hasattr(s, "model_dump") else (s if isinstance(s, dict) else {})
                    s_type = s_dict.get("type") or "summary_text"
                    s_text = s_dict.get("text") if isinstance(s_dict, dict) else str(s)
                    summary_list.append(
                        ChatMessageContentPart(
                            type=s_type,
                            text=s_text,
                            annotations=None,
                            metadata=None,
                        )
                    )
                thinking_summary = summary_list
            elif isinstance(summary_obj, str):
                thinking_summary = [
                    ChatMessageContentPart(type="summary_text", text=summary_obj, annotations=None, metadata=None)
                ]
            else:
                thinking_summary = None

            # Extract text for display, but PRESERVE original structure for round-tripping
            if isinstance(content_obj, list):
                content_text = " ".join(filter(None, (_get(c, "text", "") for c in content_obj))) or None
            else:
                content_text = _get(content_obj, "text", None)

            # Build message_contents from content_obj when possible to preserve parts
            message_contents = None
            if isinstance(content_obj, list):
                message_contents = [
                    ChatMessageContentPart(
                        type=_get(c, "type", "thinking"),
                        text=_get(c, "text", None),
                        annotations=None,
                        metadata=None,
                    )
                    for c in content_obj
                ]
            else:
                if content_text:
                    message_contents = [
                        ChatMessageContentPart(type="thinking", text=content_text, annotations=None, metadata=None)
                    ]

            ci = ChatResponseReasoningContentItem(
                index=index,
                role=role,
                thinking_id=thinking_id,
                thinking_summary=thinking_summary,  # list[ChatMessageContentPart] | None
                thinking_signature=signature,
                thinking_status=status,
                message_contents=message_contents,
            )
            return ci

        # Function/tool calls
        if item_type in ("function_call", "custom_tool_call"):
            call_id = _get(output_item, "call_id", None)
            _id = _get(output_item, "id", None)
            name = _get(output_item, "name", None)
            arguments = _get(output_item, "arguments", None)
            inputs = _get(output_item, "input", None)

            if isinstance(arguments, str):
                try:
                    import json as _json

                    arguments = _json.loads(arguments)
                except Exception:
                    # Keep as raw string if not JSON
                    pass

            args = (
                arguments
                if isinstance(arguments, dict)
                else {
                    "raw": arguments if arguments else inputs,
                }
            )

            ci = ChatResponseToolCallContentItem(
                index=index,
                role=role,
                tool_call=ChatResponseToolCall(
                    call_id=call_id,
                    id=_id,
                    name=name,
                    arguments=args,
                    metadata={"type": item_type},
                ),
                metadata={},
            )
            return ci

        # Assistant message with text/structured output content
        if item_type in ("message"):
            contents = _get(output_item, "content", None) or []
            message_id = _get(output_item, "id", None)

            # Build ChatMessageContentPart list from provider 'content' array
            parts: list[ChatMessageContentPart] = []
            aggregate_text: list[str] = []
            for c in contents:
                c_type = _get(c, "type", None)
                if c_type in ("output_text", "text"):
                    text_val = _get(c, "text", "")
                    parts.append(
                        ChatMessageContentPart(
                            type=c_type,
                            text=text_val,
                            annotations=_get(c, "annotations", None),
                        )
                    )
                    if text_val:
                        aggregate_text.append(text_val)
                else:
                    parts.append(
                        ChatMessageContentPart(
                            type=str(c_type or "unknown"),
                            text=_get(c, "text", None),
                            metadata=_get(c, "metadata", None),
                        )
                    )

            text_joined = "".join(aggregate_text)

            # If structured output is requested, emit ONLY a StructuredOutput item (no duplicate text item)
            if structured_output_config is not None:
                parsed_data, error, post_processed = ChatResponseStructuredOutput._parse_and_validate(
                    text_joined,
                    structured_output_config,
                )
                structured_output = ChatResponseStructuredOutput(
                    config=structured_output_config,
                    structured_data=parsed_data,
                    raw_data=text_joined,  # Preserve combined text for error analysis
                    parse_error=error,
                    post_processed=post_processed,
                )
                return ChatResponseStructuredOutputContentItem(
                    index=index,
                    role=role,
                    structured_output=structured_output,
                    message_id=message_id,
                    message_contents=parts,
                )

            # Otherwise, emit a single Text item carrying original parts (no plain `text` field)
            return ChatResponseTextContentItem(
                index=index,
                role=role,
                message_id=message_id,
                message_contents=parts,
            )

        # Create GenericContentItem for unhandled types like serverside tools, mcp etc.
        # TODO_FUTURE: Improve this
        ci = ChatResponseGenericContentItem(
            index=index,
            role=role,
            metadata={"raw_item": output_item},
        )
        return ci

    @staticmethod
    def dai_response_to_provider_message(
        dai_response: ChatResponse,
        model_endpoint: AIModelEndpoint,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert ChatResponse into OpenAI Responses API input format.

        Single source of truth: always converts from Dhenara content items,
        regardless of whether provider_response is available or not.
        This ensures consistent behavior for both streaming and non-streaming.
        """
        # Always use the Dhenara content items as the source of truth
        # This works for both streaming (where provider_response=None) and non-streaming
        return OpenAIMessageConverter.dai_choice_to_provider_message(
            dai_response.choices[0] if dai_response.choices else None,
            model_endpoint=model_endpoint,
            source_provider=dai_response.provider,
        )

    @staticmethod
    def dai_choice_to_provider_message(
        choice: ChatResponseChoice,
        model_endpoint: AIModelEndpoint,
        source_provider: AIModelProviderEnum,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert ChatResponseChoice into OpenAI Responses API input format.

        Returns a list of proper SDK param types for input:
        - ResponseReasoningItemParam items (one per reasoning content item)
        - ResponseOutputMessageParam (a single message with all text content)
        - ResponseFunctionToolCallParam for tool calls

        Important: OpenAI Responses API requires that ALL reasoning items must be
        followed by a message item. So we collect all reasoning items first, then
        create a single message item with all text/structured output content.
        """
        same_provider = True if str(source_provider) == str(model_endpoint.ai_model.provider) else False

        output_items: list[ResponseReasoningItemParam | ResponseOutputMessageParam | ResponseFunctionToolCallParam] = []

        # First pass: collect all content by type
        for item in choice.contents:
            try:
                if item.type == ChatResponseContentItemType.REASONING:
                    # USE PRESERVED DATA if available for perfect round-tripping
                    param_data: dict[str, Any] = {
                        "type": "reasoning",
                    }
                    if same_provider and item.thinking_id:
                        param_data["id"] = item.thinking_id

                    # Use preserved summary structure (list[dict]) if available
                    if item.thinking_summary is not None:
                        if isinstance(item.thinking_summary, list):
                            # Convert ChatMessageContentPart list to OpenAI summary list[dict]
                            summary_list = [
                                {
                                    "type": p.type,
                                    "text": p.text,
                                }
                                for p in item.thinking_summary
                            ]
                            param_data["summary"] = summary_list
                        else:
                            logger.error(f"OpenAI: Unsupported thinking_summary type; {type(item.thinking_summary)}")
                    elif item.message_contents:
                        # May be from other providers
                        # Convert string to OpenAI format
                        param_data["summary"] = [{"type": "summary_text", "text": item.get_text()}]
                    else:
                        logger.error("OpenAI: No thinking_summary or message_contents available")
                        param_data["summary"] = [{"type": "summary_text", "text": ""}]

                    # Note: For input, 'content' is NOT typically included for reasoning items
                    # Only summary is used. encrypted_content can be included if available.
                    if item.thinking_signature and same_provider:
                        param_data["encrypted_content"] = item.thinking_signature

                    # reasoning_items.append(ResponseReasoningItemParam(**param_data))
                    output_items.append(ResponseReasoningItemParam(**param_data))

                elif item.type in (ChatResponseContentItemType.TEXT, ChatResponseContentItemType.STRUCTURED_OUTPUT):
                    # Structured output are nothing but text content in model responses
                    content = []

                    if item.message_contents:
                        # Convert ChatMessageContentPart back to plain dicts for provider
                        content = [
                            {
                                "type": p.type if same_provider else "output_text",
                                "text": p.text,
                                "annotations": p.annotations,
                            }
                            for p in item.message_contents
                        ]
                    else:
                        logger.error("OpenAI: TextContentItem has no message_contents or message_contents")
                        # Fallback:
                        content.append({"type": "output_text", "text": "", "annotations": []})

                    param_data = {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
                    if same_provider and item.message_id:
                        param_data["id"] = item.message_id
                    message_param = ResponseOutputMessageParam(**param_data)
                    output_items.append(message_param)

                elif item.type == ChatResponseContentItemType.TOOL_CALL:
                    # Include tool calls in conversation history
                    # They must appear BEFORE their corresponding function_call_output items
                    tool_call = item.tool_call

                    # Convert arguments to JSON string if it's a dict
                    import json as _json

                    args_str = (
                        _json.dumps(tool_call.arguments)
                        if isinstance(tool_call.arguments, dict)
                        else str(tool_call.arguments)
                    )

                    fn_call_param = ResponseFunctionToolCallParam(
                        # NOTE: Always pass the call_id, as it will be needed to map tool call even if provider differs
                        # But do not pass any message-ids or similar
                        call_id=tool_call.call_id,
                        id=(tool_call.id if same_provider else None),
                        type="function_call",
                        name=tool_call.name,
                        arguments=args_str,
                    )
                    output_items.append(fn_call_param)

                else:
                    logger.warning(f"OpenAI: unsupported content item type: {type(item).__name__}")
            except Exception as e:
                logger.error(f"OpenAI: Validation error for item; {e}")
                raise e

        return output_items
