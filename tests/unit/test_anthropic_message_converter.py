# Provenance: Added to improve Anthropic converter coverage (2026-01-20)

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dhenara.ai.providers.anthropic.message_converter import AnthropicMessageConverter

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
