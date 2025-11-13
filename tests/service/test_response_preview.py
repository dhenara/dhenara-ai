"""Service tests for AIModelCallResponse convenience helpers."""

import pytest

from dhenara.ai.types import AIModelCallResponse
from dhenara.ai.types.genai import ChatResponse, ChatResponseChoice, ChatResponseTextContentItem
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum, AIModelProviderEnum
from dhenara.ai.types.genai.dhenara.response import AIModelCallResponseMetaData


@pytest.mark.service
@pytest.mark.case_id("DAI-035")
def test_preview_dict_none_and_happy_path():
    """
    GIVEN AIModelCallResponse instances with and without chat responses
    WHEN preview_dict is invoked
    THEN it should return None for empty responses and omit choices when populated
    """

    empty_response = AIModelCallResponse()
    assert empty_response.preview_dict() is None

    choice = ChatResponseChoice(
        index=0,
        contents=[
            ChatResponseTextContentItem(
                index=0,
                role="assistant",
                text="Preview me",
            )
        ],
    )
    chat_response = ChatResponse(
        model="stub-model",
        provider=AIModelProviderEnum.OPEN_AI,
        api_provider=AIModelAPIProviderEnum.OPEN_AI,
        choices=[choice],
        metadata=AIModelCallResponseMetaData(streaming=False, duration_seconds=0, provider_metadata={}),
    )
    populated = AIModelCallResponse(chat_response=chat_response)

    preview = populated.preview_dict()
    assert preview is not None
    assert "choices" not in preview
    assert preview["model"] == "stub-model"
