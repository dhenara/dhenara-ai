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
            # Treat provider 'thought' text as reasoning tokens in message_contents (type="thinking").
            message_contents = None
            if text is not None:
                message_contents = [ChatMessageContentPart(type="thinking", text=text, annotations=None, metadata=None)]
            return ChatResponseReasoningContentItem(
                index=index,
                role=role,
                message_id=part_id,
                message_contents=message_contents,
                thinking_summary=None,
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
            # Preserve the entire function_response object (name + response + any extras)
            if hasattr(function_response, "model_dump"):
                fr_value = function_response.model_dump()
            elif isinstance(function_response, dict):
                fr_value = function_response
            else:
                fr_value = {
                    "name": _get(function_response, "name", None),
                    "response": _get(function_response, "response", None),
                }

            return ChatResponseGenericContentItem(
                index=index,
                role=role,
                metadata={"function_response": fr_value},
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
                parsed_data, error, post_processed = ChatResponseStructuredOutput._parse_and_validate(
                    text, structured_output_config
                )

                structured_output = ChatResponseStructuredOutput(
                    config=structured_output_config,
                    structured_data=parsed_data,
                    raw_data=text,  # Keep original response regardless of parsing
                    parse_error=error,
                    post_processed=post_processed,
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

    @staticmethod
    def dai_choice_to_provider_message(
        choice: ChatResponseChoice,
        model_endpoint: AIModelEndpoint,
        source_provider: AIModelProviderEnum,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert ChatResponseChoice to Google Gemini message format.

        Google uses 'model' role and parts array with:
        - text parts for plain content
        - thought parts (thought=True + thought_signature) for reasoning
        - function_call parts for tool calls
        - structured outputs as JSON text

        Note: Google SDK types not always available; we emit strict dict schema.
        """
        same_provider = True if str(source_provider) == str(model_endpoint.ai_model.provider) else False

        parts: list[dict[str, object]] = []

        def _replay_message_contents(
            message_contents: list[ChatMessageContentPart],
            extra_kv: dict[str, Any] | None = None,
        ) -> None:
            if extra_kv is None:
                extra_kv = {}

            for p in message_contents:
                # Plain assistant/user text
                if p.type in ("text", "output_text") and p.text is not None:
                    parts.append({"text": p.text, **extra_kv})
                # Reasoning/thinking tokens captured as message_contents
                elif p.type == "thinking" and p.text is not None:
                    # Google expects thought blocks as {text, thought: True}
                    parts.append(
                        {
                            "text": p.text,
                            "thought": True,
                            **{k: v for k, v in extra_kv.items() if k != "thought"},
                        }
                    )
                elif p.type == "inline_data" and p.metadata is not None:
                    parts.append({"inline_data": p.metadata, **extra_kv})
                else:
                    # Fallback: serialize unknown parts as text
                    try:
                        parts.append({"text": json.dumps(p.model_dump()), **extra_kv})
                    except Exception:
                        parts.append({"text": str(p), **extra_kv})

        for content in choice.contents:
            if content.type == ChatResponseContentItemType.REASONING:
                # Prefer explicit reasoning message_contents (type="thinking").
                if content.message_contents:
                    parts.extend(
                        [
                            {
                                "text": p.text,
                                "thought": True,
                                "thought_signature": content.thinking_signature,
                            }
                            for p in content.message_contents
                            # if p.type in ("thinking", "text") and p.text is not None
                        ]
                    )
                    continue

                # Fallback: if only summary available, replay as thought text
                if isinstance(content.thinking_summary, list):
                    _replay_message_contents(
                        content.thinking_summary,
                        extra_kv={
                            "thought": True,
                        },
                    )
                    continue

                # Last resort empty thought block to preserve structure
                parts.append({"text": "", "thought": True})
                continue

            elif content.type == ChatResponseContentItemType.TOOL_CALL:
                tool_call = content.tool_call
                # Preserve Google thought_signature on function_call parts when continuing with the same provider
                thought_sig = None
                if same_provider and content.metadata is not None:
                    thought_sig = content.metadata.get("thought_signature")

                part_obj: dict[str, Any] = {
                    "function_call": {
                        "name": tool_call.name,
                        "args": tool_call.arguments,
                    },
                }
                if thought_sig:
                    part_obj["thought_signature"] = thought_sig

                parts.append(part_obj)
                continue
            elif content.type == ChatResponseContentItemType.TEXT:
                if content.message_contents:
                    _replay_message_contents(content.message_contents)
                else:
                    logger.warning("GoogleMessageConverter: TextContentItem has no message_contents;")
                continue
            elif content.type == ChatResponseContentItemType.STRUCTURED_OUTPUT:
                if content.message_contents:
                    _replay_message_contents(content.message_contents)
                else:
                    output = content.structured_output
                    if output and output.structured_data is not None:
                        parts.append({"text": json.dumps(output.structured_data)})
                continue

            elif content.type == ChatResponseContentItemType.GENERIC:
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
                    fr = md.get("function_response")
                    # Normalize into Google's expected shape { function_response: { name, response } }
                    if isinstance(fr, dict) and ("name" in fr or "response" in fr):
                        parts.append({"function_response": {"name": fr.get("name"), "response": fr.get("response")}})
                    else:
                        parts.append({"function_response": fr})
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
        model_endpoint: AIModelEndpoint,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert a complete ChatResponse to Google provider message format.

        Uses the first choice and relies on dai_choice_to_provider_message.
        """
        if not dai_response or not dai_response.choices:
            return {"role": "model", "parts": [{"text": ""}]}
        return GoogleMessageConverter.dai_choice_to_provider_message(
            dai_response.choices[0],
            model_endpoint=model_endpoint,
            source_provider=dai_response.provider,
        )
