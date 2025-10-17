from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import Any

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
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response import ChatResponse, ChatResponseChoice

logger = logging.getLogger(__name__)


# Helper to coerce dict-like access from SDK objects
def _get(obj: object, attr: str, default=None):
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


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
        # Google's part.thought=True indicates a thinking part with text as summary
        thought = _get(part, "thought", default=False)
        video_metadata = _get(part, "video_metadata", None)
        inline_data = _get(part, "inline_data", None)
        file_data = _get(part, "file_data", None)
        thought_signature = _get(part, "thought_signature", None)
        function_call = _get(part, "function_call", None)
        code_execution_result = _get(part, "code_execution_result", None)
        executable_code = _get(part, "executable_code", None)
        function_response = _get(part, "function_response", None)
        text = _get(part, "text", None)

        part_id = _get(part, "id", None)  # NOTE  `id` NOT available in google now
        _part_dict = part.model_dump() if hasattr(part, "model_dump") else part if isinstance(part, dict) else str(part)

        # Decode thought signature to base64 string if it is bytes-like
        if thought_signature and not isinstance(thought_signature, str):
            try:
                thought_signature = base64.b64encode(thought_signature).decode("utf-8")
            except Exception:
                # Keep as-is if encoding fails
                pass

        if thought:
            return ChatResponseReasoningContentItem(
                index=index,
                role=role,
                thinking_summary=text,
                thinking_signature=thought_signature,
                thinking_id=part_id,
            )
        if video_metadata is not None:
            return ChatResponseGenericContentItem(
                index=index,
                role=role,
                metadata={"video_metadata": video_metadata},
            )
        if inline_data is not None:
            return ChatResponseGenericContentItem(
                index=index,
                role=role,
                metadata={"inline_data": _part_dict},
            )
        if file_data is not None:
            return ChatResponseGenericContentItem(
                index=index,
                role=role,
                metadata={"file_data": file_data},
            )
        if executable_code is not None:
            return ChatResponseGenericContentItem(
                index=index,
                role=role,
                metadata={"executable_code": executable_code},
            )
        if code_execution_result is not None:
            return ChatResponseGenericContentItem(
                index=index,
                role=role,
                metadata={"code_execution_result": code_execution_result},
            )
        if function_response is not None:
            response_payload = function_response.response if hasattr(function_response, "response") else None
            return ChatResponseGenericContentItem(
                index=index,
                role=role,
                metadata={"function_response": response_payload},
            )

        if function_call is not None:
            function_payload = (
                part.function_call.model_dump() if hasattr(part.function_call, "model_dump") else part.function_call
            )
            _args = function_payload.get("args")
            _parsed_args = ChatResponseToolCall.parse_args_str_or_dict(_args)

            # TODO_FUTURE: Update fn call id when google adds it
            tool_id = function_payload.get("id")
            if tool_id is None:
                tool_id = f"dai_fncall_{uuid.uuid4().hex[:24]}"

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
                # Google sends thought_signature ONLY with function calls
                metadata={"thought_signature": thought_signature} if thought_signature else {},
            )

        # Plain text (after handling special cases). Apply structured output if requested.
        if text is not None:
            if structured_output_config is not None:
                parsed_data, error = ChatResponseStructuredOutput._parse_and_validate(text, structured_output_config)

                structured_output = ChatResponseStructuredOutput(
                    config=structured_output_config,
                    structured_data=parsed_data,
                    raw_data=text,  # Keep original response regardless of parsing
                    parse_error=error,
                )

                return ChatResponseStructuredOutputContentItem(
                    index=index,
                    role=role,
                    structured_output=structured_output,
                    message_id=part_id,
                    message_contents=[
                        ChatMessageContentPart(
                            type="text",
                            text=text,
                            annotations=None,
                            metadata=None,
                        )
                    ],
                )

            return ChatResponseTextContentItem(
                index=index,
                role=role,
                text=text,
                message_id=part_id,
                message_contents=[
                    ChatMessageContentPart(
                        type="text",
                        text=text,
                        annotations=None,
                        metadata=None,
                    )
                ],
            )

        # Fallback: represent unknown part as GENERIC for diagnostics
        return ChatResponseGenericContentItem(
            index=index,
            role=role,
            metadata={"part": _part_dict},
        )

    @staticmethod
    def provider_message_to_dai_content_items(
        *,
        message: Any,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        parts = _get(message, "parts", [])
        role = _get(message, "role", "model")
        return [
            GoogleMessageConverter.provider_part_to_content_item(
                part=part,
                index=index,
                role=role,
                structured_output_config=structured_output_config,
            )
            for index, part in enumerate(parts)
        ]

    """
    @staticmethod
    def provider_message_to_dai_content_items(
        *,
        message: Any,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        parts = getattr(message, "parts", []) or []
        role = getattr(message, "role", "model")

        items: list[ChatResponseContentItem] = []
        grouped_parts: list[ChatMessageContentPart] = []
        # Preserve provider message id if available
        message_id = getattr(message, "id", None)

        TODO: DELETE
        for index, part in enumerate(parts):

            # Thinking parts are separate content items
            if hasattr(part, "thought") and part.thought is True:
                thinking_summary = getattr(part, "text", None)
                thought_signature = getattr(part, "thought_signature", None)
                items.append(
                    ChatResponseReasoningContentItem(
                        index=index,
                        role=role,
                        thinking_summary=thinking_summary,
                        thinking_signature=thought_signature,
                        thinking_id=getattr(part, "id", None),
                    )
                )
                continue

            # Function/tool calls and their responses remain separate items
            if hasattr(part, "function_call") and part.function_call is not None:
                items.append(
                    GoogleMessageConverter.provider_part_to_content_item(
                        part=part,
                        index=index,
                        role=role,
                        structured_output_config=structured_output_config,
                    )
                )
                continue

            if hasattr(part, "function_response") and part.function_response is not None:
                items.append(
                    GoogleMessageConverter.provider_part_to_content_item(
                        part=part,
                        index=index,
                        role=role,
                        structured_output_config=structured_output_config,
                    )
                )
                continue

            # Aggregate regular message parts for round-trip preservation
            try:
                if hasattr(part, "text") and part.text is not None:
                    grouped_parts.append(
                        ChatMessageContentPart(
                            type="text",
                            text=part.text,
                            annotations=None,
                            metadata=None,
                        )
                    )
                elif hasattr(part, "inline_data") and part.inline_data is not None:
                    inline = part.inline_data
                    # inline has fields: data (bytes-like) and mime_type
                    grouped_parts.append(
                        ChatMessageContentPart(
                            type="inline_data",
                            text=None,
                            annotations=None,
                            metadata={
                                "mime_type": getattr(inline, "mime_type", None),
                                "data": getattr(inline, "data", None),
                            },
                        )
                    )
                else:
                    # Preserve unknown parts
                    grouped_parts.append(
                        ChatMessageContentPart(
                            type="unknown",
                            text=None,
                            annotations=None,
                            metadata={"raw_part": part.model_dump() if hasattr(part, "model_dump") else {}},
                        )
                    )
            except Exception:
                grouped_parts.append(
                    ChatMessageContentPart(
                        type="unknown",
                        text=None,
                        annotations=None,
                        metadata={"raw_part": str(part)},
                    )
                )

        # After iterating, emit a single message item if any grouped parts exist
        if grouped_parts:
            if structured_output_config is not None:
                # Extract concatenated text for structured output parsing
                text_value = "".join([p.text or "" for p in grouped_parts if p.type in ("text", "output_text")])
                parsed_data, error = ChatResponseStructuredOutput._parse_and_validate(
                    text_value, structured_output_config
                )

                structured_output = ChatResponseStructuredOutput(
                    config=structured_output_config,
                    structured_data=parsed_data,
                    raw_data=text_value,
                    parse_error=error,
                )

                items.append(
                    ChatResponseStructuredOutputContentItem(
                        index=len(items),
                        role=role,
                        structured_output=structured_output,
                        message_id=message_id,
                        message_contents=grouped_parts,
                    )
                )
            else:
                items.append(
                    ChatResponseTextContentItem(
                        index=len(items),
                        role=role,
                        text=None,
                        message_id=message_id,
                        message_contents=grouped_parts,
                    )
                )

        return items
    """

    @staticmethod
    def dai_choice_to_provider_message(
        choice: ChatResponseChoice,
        model_endpoint: object | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert ChatResponseChoice to Google Gemini message format.

        Google uses 'model' role and parts array with:
        - text parts for plain content
        - thought parts (thought=True + thought_signature) for reasoning
        - function_call parts for tool calls
        - structured outputs as JSON text

        Note: Google SDK types not always available; we emit strict dict schema.
        """
        parts: list[dict[str, object]] = []

        def _replay_message_contents(
            message_contents: list[ChatMessageContentPart],
            extra_kv: dict[str, Any] | None = None,
        ) -> None:
            if extra_kv is None:
                extra_kv = {}

            for p in message_contents:
                if p.type in ("text", "output_text") and p.text is not None:
                    parts.append({"text": p.text, **extra_kv})
                elif p.type == "inline_data" and p.metadata is not None:
                    parts.append({"inline_data": p.metadata, **extra_kv})
                else:
                    # Fallback: serialize unknown parts as text
                    try:
                        parts.append({"text": json.dumps(p.model_dump()), **extra_kv})
                    except Exception:
                        parts.append({"text": str(p), **extra_kv})

        for content in choice.contents:
            if isinstance(content, ChatResponseTextContentItem):
                if content.message_contents:
                    _replay_message_contents(content.message_contents)
                elif content.text is not None:
                    parts.append({"text": content.text})
                continue

            if isinstance(content, ChatResponseReasoningContentItem):
                # Prefer preserved summary text if string, else fallback to thinking_text
                text_content = None
                if isinstance(content.thinking_summary, str):
                    text_content = content.thinking_summary
                elif isinstance(content.thinking_summary, list):  # list of ChatMessageContentPart
                    _replay_message_contents(
                        content.thinking_summary,
                        extra_kv={
                            # "thought_signature": content.thinking_signature,
                            "thought": True,
                        },
                    )
                    continue
                elif content.thinking_text is not None:
                    text_content = content.thinking_text

                # NOTE: Google sends thought_signature ONLY with function calls
                parts.append(
                    {
                        "text": text_content or "",
                        "thought": True,
                        "thought_signature": content.thinking_signature,
                    }
                )

                continue

            if isinstance(content, ChatResponseToolCallContentItem) and content.tool_call is not None:
                tool_call = content.tool_call
                parts.append(
                    {
                        "function_call": {
                            "name": tool_call.name,
                            "args": tool_call.arguments,
                        },
                        "thought_signature": _get(content.metadata, "thought_signature", None),
                    }
                )
                continue

            if isinstance(content, ChatResponseStructuredOutputContentItem):
                if content.message_contents:
                    _replay_message_contents(content.message_contents)
                else:
                    output = content.structured_output
                    if output and output.structured_data is not None:
                        parts.append({"text": json.dumps(output.structured_data)})
                continue

            if isinstance(content, ChatResponseGenericContentItem):
                md = content.metadata or {}
                if "video_metadata" in md:
                    parts.append({"video_metadata": md.get("video_metadata")})
                elif "inline_data" in md:
                    parts.append({"inline_data": md.get("inline_data")})
                elif "file_data" in md:
                    parts.append({"file_data": md.get("file_data")})
                elif "executable_code" in md:
                    parts.append({"executable_code": md.get("executable_code")})
                elif "code_execution_result" in md:
                    parts.append({"code_execution_result": md.get("code_execution_result")})
                elif "function_response" in md:
                    parts.append({"function_response": md.get("function_response")})
                else:
                    parts.append({"text": json.dumps(md)})
                continue

            logger.debug(f"Google: Skipped unsupported content type {type(content)}")

        if not parts:
            parts = [{"text": ""}]

        return {"role": "model", "parts": parts}

    @staticmethod
    def dai_response_to_provider_message(
        dai_response: ChatResponse,
        model_endpoint: object | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert a complete ChatResponse to Google provider message format.

        Uses the first choice and relies on dai_choice_to_provider_message.
        """
        if not dai_response or not dai_response.choices:
            return {"role": "model", "parts": [{"text": ""}]}
        return GoogleMessageConverter.dai_choice_to_provider_message(
            dai_response.choices[0],
            model_endpoint=model_endpoint,
        )
