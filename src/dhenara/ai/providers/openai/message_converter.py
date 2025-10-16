"""Utilities for converting between OpenAI chat formats and Dhenara message types."""

from __future__ import annotations

import json
import logging

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_output_message_param import (
    ResponseOutputMessageParam,
)
from openai.types.responses.response_output_text_param import ResponseOutputTextParam
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
)

from dhenara.ai.providers.base import BaseMessageConverter
from dhenara.ai.providers.openai.constants import OPENAI_USE_RESPONSES_DEFAULT
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

    @staticmethod
    def dai_choice_to_provider_message(
        choice: ChatResponseChoice,
        *,
        model: str | None = None,
        provider: AIModelProviderEnum | None = None,
        strict_same_provider: bool = False,
    ) -> dict[str, object]:
        """Convert ChatResponseChoice into OpenAI Responses API input format.

        Returns a dict containing role='assistant' and 'output' array with proper SDK param types:
        - ResponseOutputMessageParam for text/structured outputs
        - ResponseReasoningItemParam for thinking/reasoning content
        - ResponseFunctionToolCallParam for tool calls

        This maintains full fidelity with Responses API schema, preserving reasoning IDs,
        signatures, summaries, and tool call metadata.
        """
        output_items: list[ResponseOutputMessageParam | ResponseReasoningItemParam | ResponseFunctionToolCallParam] = []

        # Group content by type to build output items
        for item in choice.contents:
            if isinstance(item, ChatResponseTextContentItem) and item.text:
                # ResponseOutputMessage with output_text content
                output_items.append(
                    ResponseOutputMessageParam(
                        id=f"msg_{len(output_items)}",  # Generate ID if not tracked
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[ResponseOutputTextParam(type="output_text", text=item.text, annotations=[])],
                    )
                )
            elif isinstance(item, ChatResponseStructuredOutputContentItem):
                output = item.structured_output
                if output and output.structured_data is not None:
                    # Structured output as output_text JSON
                    output_items.append(
                        ResponseOutputMessageParam(
                            id=f"msg_{len(output_items)}",
                            type="message",
                            role="assistant",
                            status="completed",
                            content=[
                                ResponseOutputTextParam(
                                    type="output_text", text=json.dumps(output.structured_data), annotations=[]
                                )
                            ],
                        )
                    )
            elif isinstance(item, ChatResponseReasoningContentItem):
                # ResponseReasoningItem with thinking content and summary
                thinking_id = item.thinking_id or f"reasoning_{len(output_items)}"
                summary_text = item.thinking_summary or ""
                content_parts = []
                if item.thinking_text:
                    # content is optional per SDK, omit if none
                    content_parts = [
                        ResponseOutputTextParam(type="output_text", text=item.thinking_text, annotations=[])
                    ]
                output_items.append(
                    ResponseReasoningItemParam(
                        id=thinking_id,
                        type="reasoning",
                        summary=[ResponseOutputTextParam(type="output_text", text=summary_text, annotations=[])],
                        content=content_parts or None,
                        encrypted_content=item.thinking_signature,
                        status=item.thinking_status or "completed",
                    )
                )
            elif isinstance(item, ChatResponseToolCallContentItem) and item.tool_call:
                tc = item.tool_call
                output_items.append(
                    ResponseFunctionToolCallParam(
                        id=tc.id or f"call_{len(output_items)}",
                        type="function_call",
                        call_id=tc.id or f"call_{len(output_items)}",
                        name=tc.name,
                        arguments=json.dumps(tc.arguments) if tc.arguments else "{}",
                        status="completed",
                    )
                )

        # If no output items, add an empty message to satisfy schema
        if not output_items:
            output_items.append(
                ResponseOutputMessageParam(
                    id="msg_0",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputTextParam(type="output_text", text="", annotations=[])],
                )
            )

        return {"role": "assistant", "output": output_items}
