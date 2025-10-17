"""Utilities for converting between OpenAI chat formats and Dhenara message types."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from openai.types.chat import ChatCompletionMessage
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_output_message_param import (
    ResponseOutputMessageParam,
)
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
)

from dhenara.ai.providers.base import BaseMessageConverter
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
        """Convert a single Responses API output item into ChatResponseContentItems.

        Handles item types like 'message' (with output_text items), 'reasoning', and 'function_call'.
        For 'message' content, this will also parse structured output when a schema is provided.
        """
        items: list[ChatResponseContentItem] = []
        output_item = message

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
                index=index_start,
                role=role,
                thinking_id=thinking_id,
                thinking_text=content_text,
                thinking_summary=thinking_summary,  # Preserved original structure
                thinking_signature=signature,
                thinking_status=status,
            )
            items.append(ci)

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
                index=index_start,
                role=role,
                tool_call=ChatResponseToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments if isinstance(arguments, dict) else {"raw": arguments},
                ),
                metadata={},
            )
            items.append(ci)

        # Assistant message with text/structured output content
        if item_type in ("message", "output_message"):
            contents = _get(output_item, "content", None) or []
            message_id = _get(output_item, "id", None)

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

            # IMPORTANT: Preserve the original content array for round-tripping
            # Convert Pydantic models to dicts if needed
            content_array = []
            for part in contents or []:
                if hasattr(part, "model_dump"):
                    content_array.append(part.model_dump())
                elif isinstance(part, dict):
                    content_array.append(part)
                else:
                    content_array.append({"type": "output_text", "text": str(part), "annotations": []})

            if structured_output_config is not None and text_combined:
                structured_output = ChatResponseStructuredOutput.from_model_output(
                    raw_response=text_combined,
                    config=structured_output_config,
                )
                ci = ChatResponseStructuredOutputContentItem(
                    index=index_start,
                    role=role,
                    structured_output=structured_output,
                )
                items.append(ci)
            elif text_combined is not None:
                ci = ChatResponseTextContentItem(
                    index=index_start,
                    role=role,
                    text=text_combined,
                    message_id=message_id,  # Preserved for round-tripping
                    message_content=content_array,  # Preserved for round-tripping
                )
                items.append(ci)

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
    def dai_response_to_provider_message(
        dai_response: ChatResponse,
        model_endpoint: object | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert ChatResponse into OpenAI Responses API input format."""

        # Derive strict behavior: enforce strict if same provider and not globally disabled
        is_same_provider = False
        if model_endpoint is not None:
            try:
                is_same_provider = getattr(model_endpoint.ai_model, "provider", None) == AIModelProviderEnum.OPEN_AI
            except Exception:
                is_same_provider = False
        # strict_same_provider = is_same_provider and not BaseMessageConverter.STRICT_SAME_PROVIDER_OFF

        if is_same_provider:
            original_provider_response = dai_response.provider_response
            output_items = None

            # Check if provider_response exists and has output field
            # For streaming responses, provider_response will be None
            if original_provider_response is not None:
                if hasattr(original_provider_response, "output"):
                    output_items = original_provider_response.output
                elif isinstance(original_provider_response, dict) and "output" in original_provider_response:
                    output_items = original_provider_response["output"]

            # If we have output_items from provider_response, convert them
            if output_items is not None:
                # Convert output items to proper Pydantic models, filtering out input-incompatible fields
                converted_items = []
                for item in output_items:
                    # If item is already a Pydantic model instance (ResponseReasoningItem, ResponseOutputMessage)
                    # Check using hasattr since ResponseFunctionToolCallParam is a TypedDict
                    if isinstance(item, (ResponseReasoningItem, ResponseOutputMessage)):
                        # Convert output models to input param models
                        if isinstance(item, ResponseReasoningItem):
                            param_data = {
                                "id": item.id,
                                "type": "reasoning",
                                "summary": item.summary,
                            }
                            if item.content:
                                param_data["content"] = item.content
                            if item.encrypted_content:
                                param_data["encrypted_content"] = item.encrypted_content
                            converted_items.append(ResponseReasoningItemParam(**param_data))
                        elif isinstance(item, ResponseOutputMessage):
                            param_data = {
                                "id": item.id,
                                "type": "message",
                                "role": item.role,
                                "content": item.content,
                            }
                            converted_items.append(ResponseOutputMessageParam(**param_data))
                    # If item is a dict, convert to appropriate Pydantic model
                    elif isinstance(item, dict):
                        item_type = item.get("type")

                        if item_type == "reasoning":
                            # Remove fields not allowed in input schema
                            reasoning_data = {
                                "id": item.get("id"),
                                "type": "reasoning",
                                "summary": item.get("summary"),
                            }
                            # Only include encrypted_content if present
                            # Note: 'content' is typically NOT included in reasoning input items
                            # Note: 'status' is NOT included as it's output-only
                            if item.get("encrypted_content"):
                                reasoning_data["encrypted_content"] = item["encrypted_content"]

                            converted_items.append(ResponseReasoningItemParam(**reasoning_data))

                        elif item_type == "message":
                            # Remove fields not allowed in input schema
                            message_data = {
                                "id": item.get("id"),
                                "type": "message",
                                "role": item.get("role", "assistant"),
                                "content": item.get("content", []),
                            }
                            # Note: 'status' is NOT included as it's output-only

                            converted_items.append(ResponseOutputMessageParam(**message_data))

                        elif item_type in ("function_call", "tool_call"):
                            # Convert to ResponseFunctionToolCallParam (TypedDict - can't use isinstance)
                            # Just ensure it has the required fields
                            converted_items.append(ResponseFunctionToolCallParam(**item))
                        else:
                            logger.warning(f"Unknown output item type: {item_type}, keeping as-is")
                            converted_items.append(item)
                    else:
                        # Unknown type, keep as-is
                        converted_items.append(item)

                return converted_items

        # Fallback: convert from ChatResponseChoice (used for streaming or when provider_response is None)
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

        # Track message IDs from preserved data
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
                            # Dict or other format
                            param_data["summary"] = [{"type": "summary_text", "text": str(item.thinking_summary)}]
                    else:
                        # Fallback to empty summary
                        param_data["summary"] = [{"type": "summary_text", "text": ""}]

                    # Note: For input, 'content' is NOT typically included for reasoning items
                    # Only summary is used. encrypted_content can be included if available.
                    if item.thinking_signature:
                        param_data["encrypted_content"] = item.thinking_signature

                    reasoning_items.append(ResponseReasoningItemParam(**param_data))

                elif isinstance(item, ChatResponseTextContentItem):
                    # USE PRESERVED DATA if available for perfect round-tripping
                    if item.message_id:
                        message_id_from_content = item.message_id

                    if item.message_content:
                        # Use preserved content array directly
                        preserved_content_array.extend(item.message_content)
                    else:
                        # Fallback: construct from text
                        text_contents.append({"type": "output_text", "text": item.text, "annotations": []})

                elif isinstance(item, ChatResponseToolCallContentItem):
                    # TODO: implement tool call conversion
                    pass
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
                logger.warning(f"OpenAI: Validation error for item; {e}")

        # Second pass: construct output items
        # Order is important: reasoning items must come before the message
        output_items: list[ResponseReasoningItemParam | ResponseOutputMessageParam] = []

        # Add all reasoning items first
        output_items.extend(reasoning_items)

        # Then add a single message with all text content
        # This is REQUIRED if there are any reasoning items
        if text_contents or preserved_content_array or reasoning_items:
            # Use preserved message ID if available, otherwise generate a unique one
            if message_id_from_content:
                msg_id = message_id_from_content
            else:
                # Generate unique ID using timestamp + random component
                msg_id = f"msg_{int(time.time())}_{random.randint(1000, 9999)}"

            # Use preserved content array if available, otherwise use collected text_contents
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

        # Add tool calls (if any)
        output_items.extend(tool_calls)

        return output_items
