from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dhenara.ai.types.genai.dhenara.request import Prompt
from dhenara.ai.types.genai.dhenara.request.data._tool_result import ToolCallResult, ToolCallResultsMessage
from dhenara.ai.types.genai.dhenara.response import ChatResponse


def _ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    # Dhenara Prompt
    if isinstance(value, Prompt):
        return value.get_formatted_text()

    # Pydantic models (Prompt, ChatResponse, tool results, etc.)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except TypeError:
            dumped = value.model_dump()
        return str(dumped)

    if isinstance(value, dict):
        # Common legacy/provider formats
        if "content" in value and isinstance(value["content"], str):
            return value["content"]
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        if "parts" in value and isinstance(value["parts"], list) and value["parts"]:
            part0 = value["parts"][0]
            if isinstance(part0, dict) and isinstance(part0.get("text"), str):
                return part0["text"]
        return str(value)

    return str(value)


def _message_item_to_text(message_item: Any) -> str:
    if message_item is None:
        return ""
    if isinstance(message_item, Prompt):
        return message_item.get_formatted_text()
    if isinstance(message_item, ChatResponse):
        return message_item.as_text() or ""
    if isinstance(message_item, ToolCallResult):
        return message_item.as_text()
    if isinstance(message_item, ToolCallResultsMessage):
        return "\n".join([r.as_text() for r in message_item.results])

    return _ensure_text(message_item)


def build_image_prompt_text(
    *,
    prompt: Any,
    context: Sequence[Any] | None,
    instructions: Any,
    messages: Sequence[Any] | None,
    formatter: Any | None = None,
) -> str:
    """Build a single prompt string for image-generation providers.

    Image generation endpoints typically accept only a single text prompt.
    This utility supports both legacy prompt/context/instructions and the
    newer `messages` format.

    - If `messages` is provided, it is converted into joined text.
    - Otherwise, `context` + `prompt` is used.

    `instructions` may be:
    - list[str|dict|SystemInstruction] (Dhenara format)
    - dict (provider/legacy format)

    If a formatter is provided, it will be used to join instruction lists.
    """

    instructions_str = ""
    if instructions:
        if isinstance(instructions, dict):
            instructions_str = _ensure_text(instructions)
        else:
            if formatter is None:
                instructions_str = "\n".join([_ensure_text(i) for i in instructions])
            else:
                instructions_str = formatter.join_instructions(instructions)

    if messages is not None:
        message_texts = [_message_item_to_text(mi) for mi in messages]
        messages_text = "\n".join([t for t in message_texts if t])
        prompt_text = "\n".join([t for t in [instructions_str, messages_text] if t]).strip()
    else:
        ctx_text = "\n".join([_ensure_text(c) for c in (context or []) if _ensure_text(c)])
        pr_text = _ensure_text(prompt)
        prompt_text = "\n".join([t for t in [instructions_str, ctx_text, pr_text] if t]).strip()

    if not prompt_text:
        raise ValueError("Image generation requires either `messages` or a non-empty `prompt`.")

    return prompt_text
