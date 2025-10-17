"""Message items for type-safe multi-turn conversations.

This module defines the union type for message items that can be used
in the messages input parameter. It combines:
- Prompt: for new user/system messages
- ChatResponseChoice: for previous assistant responses (contains all content items from a single turn)
- ToolCallResult: for individual tool execution results
- ToolCallResultsMessage: for grouped tool results emitted after an assistant turn
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ._prompt import Prompt
from ._tool_result import ToolCallResult, ToolCallResultsMessage

if TYPE_CHECKING:
    from dhenara.ai.types.genai.dhenara.response import ChatResponse
# MessageItem is a type-safe union of all valid message items
# that can be passed to the messages parameter
# Using Union with string forward reference to avoid circular import at runtime
# ChatResponseChoice keeps all content items (text, tool calls, etc.) together as a single assistant message
MessageItem = Union[Prompt, "ChatResponse", ToolCallResult, ToolCallResultsMessage]
