# Provenance: Added to improve message-text normalization coverage (2026-01-21)

from __future__ import annotations

import pytest

from dhenara.ai.providers.common.message_text import build_image_prompt_text
from dhenara.ai.types.genai import (
    ChatMessageContentPart,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseTextContentItem,
)
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum
from dhenara.ai.types.genai.dhenara.request.data._tool_result import ToolCallResult, ToolCallResultsMessage
from dhenara.ai.types.genai.dhenara.response import AIModelCallResponseMetaData

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-068")
def test_dai_068_image_prompt_text_accepts_tool_results_and_chat_response(text_endpoint, default_call_config):
    """GIVEN messages containing tool results, ChatResponse, and legacy provider dict shapes
    WHEN build_image_prompt_text is called in messages mode
    THEN it extracts useful text across all supported shapes.
    """

    message_part = ChatMessageContentPart(type="output_text", text="assistant-text", annotations=None)
    choice = ChatResponseChoice(
        index=0,
        contents=[
            ChatResponseTextContentItem(
                index=0,
                role="assistant",
                message_contents=[message_part],
            )
        ],
    )
    chat = ChatResponse(
        model=text_endpoint.ai_model.model_name,
        provider=text_endpoint.ai_model.provider,
        api_provider=AIModelAPIProviderEnum.OPEN_AI,
        usage=None,
        usage_charge=None,
        choices=[choice],
        metadata=AIModelCallResponseMetaData(streaming=False, duration_seconds=0, provider_metadata={}),
        provider_response={"text": "assistant-text"},
    )

    tool_results = ToolCallResultsMessage(
        results=[
            ToolCallResult(call_id="c1", output={"ok": True}),
            ToolCallResult(call_id="c2", output="done"),
        ]
    )

    out = build_image_prompt_text(
        prompt=None,
        context=None,
        instructions=None,
        messages=[
            {"parts": [{"text": "legacy-part-text"}]},
            tool_results,
            chat,
        ],
    )

    assert "legacy-part-text" in out
    assert "assistant-text" in out
    assert "done" in out
    assert "ok" in out
