# Provenance: Added to improve OpenAI converter coverage (2026-01-20)

from __future__ import annotations

import pytest

from dhenara.ai.providers.openai.message_converter import OpenAIMessageConverter
from dhenara.ai.types.genai.ai_model import AIModelAPI, AIModelAPIProviderEnum, AIModelEndpoint, AIModelProviderEnum
from dhenara.ai.types.genai.dhenara.response import ChatResponseChoice
from dhenara.ai.types.genai.dhenara.response._content_items._chat_items import ChatResponseToolCallContentItem
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT5Nano

pytestmark = [pytest.mark.component]


@pytest.mark.case_id("DAI-044")
def test_dai_044_openai_message_converter_tool_roundtrip():
    """GIVEN an OpenAI Responses output list containing a function_call
    WHEN converted to Dhenara content items and then back to provider input items
    THEN tool call arguments are preserved as a dict.
    """

    output_items = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "id": "tool_1",
            "name": "file_read",
            "arguments": '{"path": "README.md"}',
        }
    ]

    dai_items = OpenAIMessageConverter.provider_message_to_dai_content_items(message=output_items)
    assert len(dai_items) == 1
    assert isinstance(dai_items[0], ChatResponseToolCallContentItem)
    assert dai_items[0].tool_call.name == "file_read"
    assert dai_items[0].tool_call.arguments["path"] == "README.md"

    # Convert back to provider message items (Responses API input format)
    api = AIModelAPI(provider=AIModelAPIProviderEnum.OPEN_AI, api_key="sk-testkey123", credentials={}, config={})
    endpoint = AIModelEndpoint(api=api, ai_model=GPT5Nano)

    choice = ChatResponseChoice(index=0, contents=dai_items)
    provider_items = OpenAIMessageConverter.dai_choice_to_provider_message(
        choice,
        model_endpoint=endpoint,
        source_provider=AIModelProviderEnum.OPEN_AI,
    )

    # Expect at least one tool call param-like dict
    assert provider_items
    tool_items = [
        x for x in provider_items if isinstance(x, dict) and x.get("type") in ("function_call", "custom_tool_call")
    ]
    assert tool_items
    assert tool_items[0].get("name") == "file_read"
