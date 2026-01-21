# Provenance: Added to improve conversation utilities coverage (2026-01-20)

from __future__ import annotations

import pytest

from dhenara.ai.types.conversation._node import ConversationNode
from dhenara.ai.types.genai import ChatMessageContentPart
from dhenara.ai.types.genai.ai_model import AIModelProviderEnum
from dhenara.ai.types.genai.dhenara.response import ChatResponse, ChatResponseChoice
from dhenara.ai.types.genai.dhenara.response._content_items._chat_items import ChatResponseTextContentItem

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-041")
def test_dai_041_context_filters_none_and_includes_response():
    """GIVEN a ConversationNode without and with a response
    WHEN get_context() is called
    THEN it always returns a user prompt and includes the response prompt when present.
    """

    node_no_resp = ConversationNode(user_query="hello")
    ctx = node_no_resp.get_context()
    assert len(ctx) == 1
    assert ctx[0].role == "user"
    assert ctx[0].get_formatted_text() == "hello"

    resp = ChatResponse(
        model="dummy",
        provider=AIModelProviderEnum.OPEN_AI,
        choices=[
            ChatResponseChoice(
                index=0,
                contents=[
                    ChatResponseTextContentItem(
                        index=0,
                        role="assistant",
                        message_contents=[ChatMessageContentPart(type="text", text="hi", annotations=None)],
                    )
                ],
            )
        ],
    )

    node_with_resp = ConversationNode(user_query="hello", response=resp)
    ctx2 = node_with_resp.get_context()
    assert len(ctx2) == 2
    assert ctx2[0].role == "user"
    assert ctx2[1].role == "assistant"
    assert "hi" in ctx2[1].get_formatted_text()
