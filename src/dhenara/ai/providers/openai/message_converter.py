"""Utilities for converting between OpenAI chat formats and Dhenara message types."""

from __future__ import annotations

import logging
from typing import Any

from openai.types.chat import ChatCompletionMessage
from openai.types.responses import (
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)

from dhenara.ai.providers.base import BaseMessageConverter
from dhenara.ai.types.genai import (
    ChatMessageContentPart,
    ChatResponseContentItem,
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
        message: ChatCompletionMessage,
        role: str,
        index_start: int,
        ai_model_provider: AIModelProviderEnum,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        content_index = index_start
        content_items: list[ChatResponseContentItem] = []
        for item in message:  # message is `output` list
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
        message_item: ChatCompletionMessage,
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

            # Extract text for display, but PRESERVE original structure for round-tripping
            if isinstance(content_obj, list):
                content_text = " ".join(filter(None, (_get(c, "text", "") for c in content_obj))) or None
            else:
                content_text = _get(content_obj, "text", None)

            # IMPORTANT: Store thinking_summary as list[ChatMessageContentPart] | str | None
            # Convert OpenAI summary list items to ChatMessageContentPart for predictable handling
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

            ci = ChatResponseReasoningContentItem(
                index=index,
                role=role,
                thinking_id=thinking_id,
                thinking_text=content_text,
                thinking_summary=thinking_summary,  # Preserved original structure
                thinking_signature=signature,
                thinking_status=status,
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

            # IMPORTANT: Preserve the original content array for round-tripping
            # Convert Pydantic models to dicts if needed
            content_array: list[ChatMessageContentPart] = []
            for part in contents or []:
                try:
                    ctype = _get(part, "type", None)
                    if ctype == "output_text":
                        text = _get(part, "text", None)
                        annotations = _get(part, "annotations", None)
                        content_array.append(
                            ChatMessageContentPart(
                                type=ctype,
                                text=text,
                                annotations=annotations,
                            )
                        )
                        continue
                    elif ctype == "refusal":
                        reason = _get(part, "refusal", None)
                        content_array.append(
                            ChatMessageContentPart(
                                type=ctype,
                                text=None,
                                annotations=None,
                                metadata={"refusal": reason},
                            )
                        )
                        continue
                    else:
                        logger.error("OpenAI: TextContentItem has no message_contents or message_contents")
                        continue
                except Exception as _e:
                    logger.error("OpenAI: Failed to convert message part to ChatMessageContentPart")

            # Extract text from content array for structured output or text display
            if structured_output_config is not None:
                # For structured output, extract text from message content parts
                text_parts = [part.text for part in content_array if part.type == "output_text" and part.text]
                text_combined = "".join(text_parts).strip() if text_parts else None

                parsed_data, error = ChatResponseStructuredOutput._parse_and_validate(
                    raw_data=text_combined,
                    config=structured_output_config,
                )

                structured_output = ChatResponseStructuredOutput(
                    structured_data=parsed_data,
                    parse_error=error,
                    config=structured_output_config,
                    raw_data=text_combined,  # Preserve combined text for error analysis
                )

                ci = ChatResponseStructuredOutputContentItem(
                    index=index,
                    role=role,
                    structured_output=structured_output,
                    message_id=message_id,
                    message_contents=content_array,
                )
            else:
                ci = ChatResponseTextContentItem(
                    index=index,
                    role=role,
                    text=None,  # For openai, use message_contents array instead of text
                    message_id=message_id,
                    message_contents=content_array,
                )

            return ci

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
                if isinstance(item, ChatResponseReasoningContentItem):
                    # USE PRESERVED DATA if available for perfect round-tripping
                    param_data: dict[str, Any] = {
                        "type": "reasoning",
                    }
                    if same_provider and item.thinking_id:
                        param_data["id"] = item.thinking_id

                    # Use preserved summary structure (list[dict] or str) if available
                    if item.thinking_summary is not None:
                        if isinstance(item.thinking_summary, list):
                            # Convert ChatMessageContentPart list to OpenAI summary list[dict]
                            summary_list = [
                                {
                                    "type": getattr(p, "type", "summary_text"),
                                    "text": getattr(p, "text", None),
                                }
                                for p in item.thinking_summary
                            ]
                            param_data["summary"] = summary_list
                        elif isinstance(item.thinking_summary, str):
                            param_data["summary"] = [{"type": "summary_text", "text": item.thinking_summary}]
                        else:
                            logger.error(f"OpenAI: Unsupported thinking_summary type; {type(item.thinking_summary)}")
                    elif item.thinking_text is not None:
                        # May be from other providers
                        # Convert string to OpenAI format
                        param_data["summary"] = [{"type": "summary_text", "text": item.thinking_text}]
                    else:
                        logger.error("OpenAI: No thinking_summary or thinking_text available")
                        # Fallback to empty summary
                        param_data["summary"] = [{"type": "summary_text", "text": ""}]

                    # Note: For input, 'content' is NOT typically included for reasoning items
                    # Only summary is used. encrypted_content can be included if available.
                    if item.thinking_signature and same_provider:
                        param_data["encrypted_content"] = item.thinking_signature

                    # reasoning_items.append(ResponseReasoningItemParam(**param_data))
                    output_items.append(ResponseReasoningItemParam(**param_data))

                elif isinstance(item, (ChatResponseTextContentItem, ChatResponseStructuredOutputContentItem)):
                    # Structured output are nothing but text content in model responses
                    content = []

                    if item.message_contents:
                        # Convert ChatMessageContentPart back to plain dicts for provider
                        content = [
                            {
                                "type": p.type,
                                "text": p.text,
                                "annotations": p.annotations,
                            }
                            for p in item.message_contents
                        ]
                    elif item.text is not None:
                        content.append({"type": "output_text", "text": item.text, "annotations": []})
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

                elif isinstance(item, ChatResponseToolCallContentItem):
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
                        call_id=(tool_call.call_id if same_provider else None),
                        id=(tool_call.id if same_provider else None),
                        type="function_call",
                        name=tool_call.name,
                        arguments=args_str,
                    )
                    output_items.append(fn_call_param)

                else:
                    logger.warning(f"OpenAI: unsupported content item type: {type(item).__name__}")
            except Exception as e:  # noqa: PERF203
                logger.error(f"OpenAI: Validation error for item; {e}")
                raise e

        return output_items
