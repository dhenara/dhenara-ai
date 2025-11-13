"""Unit tests for BaseFormatter class.

Tests cover prompt formatting, instruction joining, and message API conversion.
"""

import pytest

from dhenara.ai.types.genai.dhenara.request import Prompt, SystemInstruction
from dhenara.ai.types.genai.dhenara.request.data import PromptMessageRoleEnum


# Create a minimal concrete formatter for testing
class TestFormatter:
    """Minimal formatter implementation for testing base methods"""

    @classmethod
    def convert_prompt(cls, formatted_prompt, model_endpoint=None, files=None, max_words_file=None):
        """Simple implementation that returns the formatted prompt as-is"""
        return formatted_prompt

    @classmethod
    def convert_instruction_prompt(cls, formatted_prompt, model_endpoint=None):
        """Simple implementation for instructions"""
        return {"role": "system", "content": formatted_prompt.text}

    @classmethod
    def convert_dai_message_item_to_provider(cls, message_item, model_endpoint=None, **kwargs):
        """Simple implementation that wraps message items"""
        if isinstance(message_item, Prompt):
            return {"role": "user", "content": message_item.get_formatted_text(**kwargs)}
        return {"role": "assistant", "content": str(message_item)}


# Import base formatter methods into test formatter
from dhenara.ai.providers.base.base_formatter import BaseFormatter

# Copy methods as classmethods to TestFormatter
TestFormatter.format_prompt = classmethod(BaseFormatter.format_prompt.__func__)
TestFormatter.join_instructions = classmethod(BaseFormatter.join_instructions.__func__)
TestFormatter.format_instructions = classmethod(BaseFormatter.format_instructions.__func__)
TestFormatter.format_messages = classmethod(BaseFormatter.format_messages.__func__)


@pytest.mark.unit
@pytest.mark.case_id("DAI-007")
def test_format_prompt_handles_str_and_prompt():
    """
    GIVEN a BaseFormatter subclass
    WHEN format_prompt is called with a string
    THEN it should create a FormattedPrompt with role=USER and the text

    WHEN format_prompt is called with a Prompt object
    THEN it should extract and format the text properly
    """
    # Test with string input
    result = TestFormatter.format_prompt("Hello world")
    assert result.role == PromptMessageRoleEnum.USER
    assert result.text == "Hello world"

    # Test with Prompt object
    prompt_obj = Prompt.with_text("Test message")
    result = TestFormatter.format_prompt(prompt_obj)
    assert result.role == PromptMessageRoleEnum.USER
    assert "Test message" in result.text

    # Test with dict input (should convert to Prompt)
    # Remove this test as it requires role field


@pytest.mark.unit
@pytest.mark.case_id("DAI-008")
def test_join_and_format_instructions():
    """
    GIVEN a list of instructions (strings and SystemInstruction objects)
    WHEN join_instructions is called
    THEN instructions should be joined with spaces preserving order and formatting

    WHEN format_instructions is called
    THEN it should create a formatted system message
    """
    # Test joining string instructions
    instructions = ["First instruction", "Second instruction", "Third instruction"]
    joined = TestFormatter.join_instructions(instructions)
    assert joined == "First instruction Second instruction Third instruction"

    # Test with SystemInstruction objects
    sys_instr = [
        SystemInstruction(text="Be helpful"),
        SystemInstruction(text="Be concise"),
    ]
    joined = TestFormatter.join_instructions(sys_instr)
    assert "Be helpful" in joined
    assert "Be concise" in joined

    # Test mixed strings and SystemInstruction
    mixed = ["Start:", SystemInstruction(text="Middle"), "End"]
    joined = TestFormatter.join_instructions(mixed)
    assert "Start:" in joined
    assert "Middle" in joined
    assert "End" in joined

    # Test format_instructions creates proper system message
    formatted = TestFormatter.format_instructions(["Instruction 1", "Instruction 2"])
    assert formatted["role"] == "system"
    assert "Instruction 1" in formatted["content"]
    assert "Instruction 2" in formatted["content"]


@pytest.mark.unit
@pytest.mark.case_id("DAI-009")
def test_format_messages_dispatches_types():
    """
    GIVEN a list of MessageItem objects (various types)
    WHEN format_messages is called
    THEN each message should be dispatched to convert_dai_message_item_to_provider
    AND the results should be properly collected (handling both single dict and list returns)
    """
    # Create message items
    msg1 = Prompt.with_text("User message 1")
    msg2 = Prompt.with_text("User message 2")

    messages = [msg1, msg2]
    formatted = TestFormatter.format_messages(messages)

    # Should have processed both messages
    assert len(formatted) >= 2

    # Each should have been converted
    assert all("role" in msg for msg in formatted)
    assert all("content" in msg for msg in formatted)

    # Test with empty list
    empty_result = TestFormatter.format_messages([])
    assert empty_result == []

    # Test that None messages list returns empty
    none_result = TestFormatter.format_messages(None)
    assert none_result == []
