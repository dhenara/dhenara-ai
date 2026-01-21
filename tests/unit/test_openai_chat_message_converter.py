from __future__ import annotations

from types import SimpleNamespace

import pytest

from dhenara.ai.providers.openai.chat_completions_api.message_converter import OpenAIMessageConverterChatCompletions
from dhenara.ai.types.genai.ai_model import AIModelProviderEnum

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-045")
def test_dai_045_chat_message_converter_basic():
    """GIVEN a ChatCompletionMessage-like object with content
    WHEN converted to content items
    THEN a text content item is produced.
    """

    msg = SimpleNamespace(content="hello", tool_calls=None)
    items = OpenAIMessageConverterChatCompletions.provider_message_to_content_items(
        message=msg,
        role="assistant",
        index_start=0,
        ai_model_provider=AIModelProviderEnum.OPEN_AI,
        structured_output_config=None,
    )

    assert items
    assert items[0].get_text() == "hello"
