# Provenance: Added to improve Anthropic converter coverage (2026-01-20)

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from dhenara.ai.providers.anthropic.message_converter import AnthropicMessageConverter
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response._content_items._chat_items import ChatResponseGenericContentItem

pytestmark = [pytest.mark.unit]


class _ToolUseBlock(SimpleNamespace):
    def model_dump(self):
        return {"type": "tool_use", "id": "call_1", "name": "file_read", "input": {"path": "README.md"}}


@pytest.mark.case_id("DAI-047")
def test_dai_047_anthropic_message_conversion_text_and_tools():
    """GIVEN Anthropic-like content blocks for text and tool_use
    WHEN converted to Dhenara content items
    THEN text becomes a text item and tool_use becomes a tool call item.
    """

    text_block = SimpleNamespace(type="text", text="hello")
    tool_block = _ToolUseBlock(type="tool_use")

    items_text = AnthropicMessageConverter._content_block_to_items(
        content_block=text_block,
        index=0,
        role="assistant",
        structured_output_config=None,
    )
    assert items_text
    assert items_text[0].get_text() == "hello"

    items_tool = AnthropicMessageConverter._content_block_to_items(
        content_block=tool_block,
        index=1,
        role="assistant",
        structured_output_config=None,
    )
    assert items_tool
    assert getattr(items_tool[0], "tool_call", None) is not None
    assert items_tool[0].tool_call.name == "file_read"
    assert items_tool[0].tool_call.arguments["path"] == "README.md"


class _TravelPlan(BaseModel):
    destination: str
    days: int
    interests: list[str]


@pytest.mark.case_id("DAI-115")
def test_dai_115_anthropic_native_structured_output_skips_whitespace_text_block():
    cfg = StructuredOutputConfig.from_model(model_class=_TravelPlan)

    # Opus 4.6 adaptive-thinking can yield an initial whitespace-only text block.
    whitespace_text_block = SimpleNamespace(type="text", text="\n\n")

    items = AnthropicMessageConverter._content_block_to_items(
        content_block=whitespace_text_block,
        index=0,
        role="assistant",
        structured_output_config=cfg,
    )

    assert items
    assert items[0].type == "text"
    assert items[0].get_text() == "\n\n"


@pytest.mark.case_id("DAI-123")
def test_dai_123_anthropic_server_tool_and_citations_are_preserved():
    text_block = SimpleNamespace(
        type="text",
        text="Claude Shannon was born in 1916.",
        citations=[{"type": "web_search_result_location", "url": "https://example.com", "title": "Example"}],
    )
    server_tool_use = SimpleNamespace(
        type="server_tool_use", id="srvtool_1", name="web_search", input={"query": "claude shannon"}
    )
    web_search_result = SimpleNamespace(
        type="web_search_tool_result",
        tool_use_id="srvtool_1",
        content=[{"type": "web_search_result", "url": "https://example.com", "title": "Example"}],
    )

    text_items = AnthropicMessageConverter._content_block_to_items(
        content_block=text_block,
        index=0,
        role="assistant",
        structured_output_config=None,
    )
    assert text_items[0].message_contents[0].annotations[0]["url"] == "https://example.com"

    server_items = AnthropicMessageConverter._content_block_to_items(
        content_block=server_tool_use,
        index=1,
        role="assistant",
        structured_output_config=None,
    )
    assert isinstance(server_items[0], ChatResponseGenericContentItem)
    assert server_items[0].metadata["server_tool_use"]["name"] == "web_search"

    result_items = AnthropicMessageConverter._content_block_to_items(
        content_block=web_search_result,
        index=2,
        role="assistant",
        structured_output_config=None,
    )
    assert isinstance(result_items[0], ChatResponseGenericContentItem)
    assert result_items[0].metadata["web_search_tool_result"]["tool_use_id"] == "srvtool_1"
