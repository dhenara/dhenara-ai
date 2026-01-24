"""Utilities for converting between OpenAI Chat Completions formats and Dhenara message types."""

from __future__ import annotations

from collections.abc import Iterable

from openai.types.chat import ChatCompletionMessage

from dhenara.ai.types.genai import (
    ChatMessageContentPart,
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
from dhenara.ai.utils.dai_disk import DAI_JSON


class OpenAIMessageConverterChatCompletions:
    """Bidirectional converter for OpenAI Chat Completions messages."""

    @staticmethod
    def provider_message_to_content_items(
        *,
        message: ChatCompletionMessage,
        role: str,
        index_start: int,
        ai_model_provider: AIModelProviderEnum,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        """Convert an OpenAI provider message into ChatResponseContentItems."""

        if getattr(message, "content", None):
            content_text = message.content
            items: list[ChatResponseContentItem] = []

            # DeepSeek reasoning separation (uses <think> tags)
            if ai_model_provider == AIModelProviderEnum.DEEPSEEK and isinstance(content_text, str):
                import re

                think_match = re.search(r"<think>(.*?)</think>", content_text, re.DOTALL)
                if think_match:
                    reasoning_content = think_match.group(1).strip()
                    if reasoning_content:
                        items.append(
                            ChatResponseReasoningContentItem(
                                index=index_start,
                                role=role,
                                message_contents=[
                                    ChatMessageContentPart(
                                        type="thinking",
                                        text=reasoning_content,
                                        annotations=None,
                                    )
                                ],
                            )
                        )
                    answer_content = re.sub(r"<think>.*?</think>", "", content_text, flags=re.DOTALL).strip()
                    content_text = answer_content or None

            if structured_output_config is not None and content_text:
                parsed_data, error, post_processed = ChatResponseStructuredOutput.parse_and_validate(
                    content_text, structured_output_config
                )

                structured_output = ChatResponseStructuredOutput(
                    config=structured_output_config,
                    structured_data=parsed_data,
                    raw_data=content_text,
                    parse_error=error,
                    post_processed=post_processed,
                )

                items.append(
                    ChatResponseStructuredOutputContentItem(
                        index=index_start,
                        role=role,
                        structured_output=structured_output,
                    )
                )
            elif content_text:
                items.append(
                    ChatResponseTextContentItem(
                        index=index_start,
                        role=role,
                        message_contents=[ChatMessageContentPart(type="text", text=content_text, annotations=None)],
                    )
                )

            return items

        if getattr(message, "tool_calls", None):
            tool_call_items: list[ChatResponseContentItem] = []
            for tool_call in message.tool_calls or []:
                if isinstance(tool_call, dict):
                    tool_payload = tool_call
                else:
                    tool_payload = tool_call.model_dump()

                _args = tool_payload.get("function", {}).get("arguments")
                _parsed_args = ChatResponseToolCall.parse_args_str_or_dict(_args if _args is not None else {})

                tool_call = ChatResponseToolCall(
                    id=tool_payload.get("id"),
                    name=(
                        tool_payload.get("function", {}).get("name")
                        if isinstance(tool_payload.get("function", {}).get("name"), str)
                        else None
                    )
                    or "unknown_tool",
                    arguments=_parsed_args.get("arguments_dict") or {},
                    metadata={},
                )

                tool_call_items.append(
                    ChatResponseToolCallContentItem(
                        index=index_start,
                        role=role,
                        tool_call=tool_call,
                        metadata={},
                    )
                )
                index_start += 1

            return tool_call_items

        # No known message content
        return []

    @staticmethod
    def convert_choice_to_provider_message(choice: ChatResponseChoice) -> dict:
        """Convert a Dhenara choice object back to an OpenAI chat message dict."""
        role = "assistant"
        content_parts: list[str] = []

        for item in choice.contents or []:
            try:
                text = item.get_text()  # type: ignore[attr-defined]
            except Exception:
                text = None
            if text:
                content_parts.append(text)

        content = "\n".join(content_parts)
        return {"role": role, "content": content}

    @staticmethod
    def parse_tool_call_args_str_or_dict(args: str | dict | None) -> dict:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        try:
            return DAI_JSON.loads(args)
        except Exception:
            return {"raw": args}

    @staticmethod
    def _flatten_iterable(iterable: Iterable[object]) -> list[object]:
        return list(iterable)
