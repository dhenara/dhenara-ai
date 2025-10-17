"""Utilities for converting between Google/Gemini chat formats and Dhenara message types."""

from __future__ import annotations

import json
import uuid
from typing import Any

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
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response import ChatResponseChoice


class GoogleMessageConverter(BaseMessageConverter):
    """Bidirectional converter for Google Gemini messages."""

    @staticmethod
    def provider_part_to_content_item(
        *,
        part: Any,
        index: int,
        role: str,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> ChatResponseContentItem:
        # Handle thinking/thought content first (Google's encrypted reasoning)
        # Google's part.thought=True indicates this is a thinking part
        # The part.text contains the thinking SUMMARY (not full reasoning - that's encrypted)
        # The part.thought_signature is an encrypted signature for multi-turn context
        if hasattr(part, "thought") and part.thought is True:
            thinking_summary = part.text if hasattr(part, "text") else None
            thought_signature = getattr(part, "thought_signature", None)
            # thought_signature base64.b64encode(part.thought_signature).decode("utf-8")

            return ChatResponseReasoningContentItem(
                index=index,
                role=role,
                thinking_summary=thinking_summary,
                thinking_signature=thought_signature,
            )

        if hasattr(part, "text") and part.text is not None:
            text_value = part.text

            if structured_output_config is not None:
                parsed_data, error = ChatResponseStructuredOutput._parse_and_validate(
                    text_value, structured_output_config
                )

                structured_output = ChatResponseStructuredOutput(
                    config=structured_output_config,
                    structured_data=parsed_data,
                    raw_data=text_value,  # Keep original response regardless of parsing
                    parse_error=error,
                )

                return ChatResponseStructuredOutputContentItem(
                    index=index,
                    role=role,
                    structured_output=structured_output,
                )

            return ChatResponseTextContentItem(
                index=index,
                role=role,
                text=text_value,
            )

        if hasattr(part, "function_call") and part.function_call is not None:
            function_payload = (
                part.function_call.model_dump() if hasattr(part.function_call, "model_dump") else part.function_call
            )
            _args = function_payload.get("args")
            _parsed_args = ChatResponseToolCall.parse_args_str_or_dict(_args)

            # TODO: REview. There should be some sort of id
            # Google sometimes doesn't provide an ID (when using AFC or in certain scenarios)
            # Generate a unique ID if not provided to ensure consistency across providers
            tool_id = function_payload.get("id")
            if tool_id is None:
                tool_id = f"call_{uuid.uuid4().hex[:24]}"

            tool_call = ChatResponseToolCall(
                call_id=tool_id,
                id=None,
                name=function_payload.get("name"),
                arguments=_parsed_args.get("arguments_dict"),
                raw_data=_parsed_args.get("raw_data"),
                parse_error=_parsed_args.get("parse_error"),
            )
            return ChatResponseToolCallContentItem(
                index=index,
                role=role,
                tool_call=tool_call,
                metadata={},
            )

        if hasattr(part, "function_response") and part.function_response is not None:
            response_payload = part.function_response.response if hasattr(part.function_response, "response") else None
            return ChatResponseTextContentItem(
                index=index,
                role=role,
                text=json.dumps(response_payload) if response_payload else "",
            )

        return ChatResponseGenericContentItem(
            index=index,
            role=role,
            metadata={"part": part.model_dump() if hasattr(part, "model_dump") else {}},
        )

    @staticmethod
    def provider_message_to_dai_content_items(
        *,
        message: Any,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        parts = getattr(message, "parts", []) or []
        role = getattr(message, "role", "model")

        return [
            GoogleMessageConverter.provider_part_to_content_item(
                part=part,
                index=index,
                role=role,
                structured_output_config=structured_output_config,
            )
            for index, part in enumerate(parts)
        ]

    @staticmethod
    def dai_choice_to_provider_message(
        choice: ChatResponseChoice,
        *,
        model: str | None = None,
        provider: str | None = None,
        strict_same_provider: bool = False,
    ) -> dict[str, object]:
        """Convert ChatResponseChoice to Google Gemini message format.

        Google uses 'model' role and parts array with:
        - text parts for plain content
        - thought parts (thought=True + thought_signature) for reasoning
        - function_call parts for tool calls
        - structured outputs as JSON text

        Note: Google SDK types not always available; we emit strict dict schema.
        """
        parts: list[dict[str, object]] = []

        for content in choice.contents:
            if isinstance(content, ChatResponseTextContentItem) and content.text:
                parts.append({"text": content.text})
            elif isinstance(content, ChatResponseReasoningContentItem):
                # Google thinking: thought=True, text=summary, thought_signature for multi-turn
                signature = content.thinking_signature
                text_content = content.thinking_summary or content.thinking_text
                if text_content and signature:
                    # Proper thought part with signature
                    parts.append({"text": text_content, "thought": True, "thought_signature": signature})
                elif text_content:
                    if strict_same_provider:
                        raise ValueError("Google: missing thought_signature for reasoning content in strict mode.")
                    # Fallback: emit as text if no signature (cross-provider compatibility)
                    parts.append({"text": text_content})
            elif isinstance(content, ChatResponseToolCallContentItem) and content.tool_call:
                tool_call = content.tool_call
                parts.append(
                    {
                        "function_call": {
                            "name": tool_call.name,
                            "args": tool_call.arguments,
                        }
                    }
                )
            elif isinstance(content, ChatResponseStructuredOutputContentItem):
                output = content.structured_output
                if output and output.structured_data:
                    parts.append({"text": json.dumps(output.structured_data)})

        if not parts:
            parts = [{"text": ""}]

        return {"role": "model", "parts": parts}
