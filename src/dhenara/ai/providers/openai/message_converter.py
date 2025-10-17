"""Utilities for converting between OpenAI chat formats and Dhenara message types."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from openai.types.chat import ChatCompletionMessage
from openai.types.responses import (
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)

from dhenara.ai.providers.base import BaseMessageConverter
from dhenara.ai.types.genai import (
    ChatResponseContentItem,
    ChatResponseGenericContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseStructuredOutput,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCall,
    ChatResponseToolCallContentItem,
)
from dhenara.ai.types.genai.ai_model import AIModelProviderEnum
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

            # IMPORTANT: Store the original summary_obj structure (list[dict] or str)
            # This allows perfect round-tripping to OpenAI API
            if isinstance(summary_obj, list):
                # Convert list of Summary Pydantic objects to list[dict]
                summary_list = []
                for s in summary_obj:
                    if hasattr(s, "model_dump"):
                        summary_list.append(s.model_dump())
                    elif isinstance(s, dict):
                        summary_list.append(s)
                    else:
                        # Fallback: treat as unknown object, try to extract text
                        summary_list.append({"type": "summary_text", "text": str(s)})
                thinking_summary = summary_list
            elif summary_obj is not None:
                # If it's a Pydantic object, convert to dict
                if hasattr(summary_obj, "model_dump"):
                    thinking_summary = summary_obj.model_dump()
                else:
                    # If it's a string or dict, keep it as-is
                    thinking_summary = summary_obj
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

            ci = ChatResponseToolCallContentItem(
                index=index,
                role=role,
                tool_call=ChatResponseToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments if isinstance(arguments, dict) else {"raw": arguments},
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
            content_array = []
            for part in contents or []:
                if hasattr(part, "model_dump"):
                    content_array.append(part.model_dump())
                elif isinstance(part, dict):
                    content_array.append(part)
                else:
                    logger.error("OpenAI: TextContentItem has no message_contents or message_contents")
                    pass

            if structured_output_config is not None:
                text_combined = "".join(content_array).strip() if content_array else None
            else:
                text_combined = None  # For openai, use message_contents array instead of text

            if structured_output_config is not None and text_combined is not None:
                structured_output = ChatResponseStructuredOutput.from_model_output(
                    raw_response=text_combined,
                    config=structured_output_config,
                )
                ci = ChatResponseStructuredOutputContentItem(
                    index=index,
                    role=role,
                    structured_output=structured_output,
                )
            else:
                ci = ChatResponseTextContentItem(
                    index=index,
                    role=role,
                    text=text_combined,
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
        model_endpoint: object | None = None,
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
        )

    @staticmethod
    def dai_choice_to_provider_message(
        choice: ChatResponseChoice,
        model_endpoint: object | None = None,
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
        reasoning_items: list[ResponseReasoningItemParam] = []
        text_contents: list[dict[str, Any]] = []
        tool_calls: list[ResponseFunctionToolCallParam] = []
        output_items: list[ResponseReasoningItemParam | ResponseOutputMessageParam | ResponseFunctionToolCallParam] = []

        # Track message IDs and content from preserved data
        message_id_from_content = None
        preserved_content_array = []

        # First pass: collect all content by type
        for item in choice.contents:
            try:
                if isinstance(item, ChatResponseReasoningContentItem):
                    # USE PRESERVED DATA if available for perfect round-tripping
                    param_data = {
                        "id": item.thinking_id or f"rs_{len(reasoning_items)}",
                        "type": "reasoning",
                    }

                    # Use preserved summary structure (list[dict] or str) if available
                    if item.thinking_summary is not None:
                        if isinstance(item.thinking_summary, list):
                            # Already in OpenAI format [{"type": "summary_text", "text": "..."}]
                            param_data["summary"] = item.thinking_summary
                        elif isinstance(item.thinking_summary, str):
                            # Convert string to OpenAI format
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
                    if item.thinking_signature:
                        param_data["encrypted_content"] = item.thinking_signature

                    # reasoning_items.append(ResponseReasoningItemParam(**param_data))
                    output_items.append(ResponseReasoningItemParam(**param_data))

                elif isinstance(item, ChatResponseTextContentItem):
                    content = []

                    if item.message_contents:
                        content = item.message_contents
                    elif item.text is not None:
                        content.append({"type": "output_text", "text": item.text, "annotations": []})
                    else:
                        logger.error("OpenAI: TextContentItem has no message_contents or message_contents")
                        # Fallback:
                        content.append({"type": "output_text", "text": "", "annotations": []})

                    param_data = {
                        "id": item.message_id or f"msg_{int(time.time())}_{random.randint(1000, 9999)}",
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
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

                    tool_call_param = ResponseFunctionToolCallParam(
                        call_id=tool_call.id,
                        type="function_call",
                        name=tool_call.name,
                        arguments=args_str,
                    )
                    output_items.append(tool_call_param)
                elif isinstance(item, ChatResponseStructuredOutputContentItem):
                    # TODO: implement structured output conversion
                    # For now, convert to text
                    if hasattr(item.structured_output, "raw_response"):
                        text_contents.append(
                            {"type": "output_text", "text": item.structured_output.raw_response, "annotations": []}
                        )
                else:
                    logger.warning(f"OpenAI: unsupported content item type: {type(item).__name__}")
            except Exception as e:  # noqa: PERF203
                logger.error(f"OpenAI: Validation error for item; {e}")
                raise e

        return output_items

        # Second pass: construct output items

        # Add all reasoning items first
        output_items.extend(reasoning_items)

        # Determine what to add next based on content
        has_text = bool(text_contents or preserved_content_array)
        has_reasoning = bool(reasoning_items)
        has_tools = bool(tool_calls)

        # Add message ONLY if:
        # 1. We have actual text content, OR
        # 2. We have reasoning but NO tool calls (reasoning alone requires message)
        # Do NOT add empty message when we have reasoning + tool calls (violates OpenAI rules)
        if has_text or (has_reasoning and not has_tools):
            # Use preserved message ID if available, otherwise generate a unique one
            if message_id_from_content:
                msg_id = message_id_from_content
            else:
                # Generate unique ID using timestamp + random component
                msg_id = f"msg_{int(time.time())}_{random.randint(1000, 9999)}"

            # Use preserved content array if available, otherwise use collected text_contents
            # If no text but reasoning exists, use empty text (required by API)
            final_content = (
                preserved_content_array
                if preserved_content_array
                else (text_contents if text_contents else [{"type": "output_text", "text": "", "annotations": []}])
            )

            message_param = ResponseOutputMessageParam(
                id=msg_id,
                type="message",
                role="assistant",
                content=final_content,
            )
            output_items.append(message_param)

        # Add tool calls after message (if any)
        # Tool calls must come AFTER message but BEFORE function_call_output
        output_items.extend(tool_calls)

        return output_items
