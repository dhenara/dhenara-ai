"""Message items for type-safe multi-turn conversations.

This module defines the union type for message items that can be used
in the messages input parameter. It combines:
- Prompt: for new user/system messages
- ChatResponseContentItem: for previous assistant responses (text, reasoning, tool calls, etc.)
- ToolCallResult: for tool execution results
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ._prompt import Prompt
from ._tool_result import ToolCallResult

if TYPE_CHECKING:
    from dhenara.ai.types.genai.dhenara.response import ChatResponseContentItem

# MessageItem is a type-safe union of all valid message items
# that can be passed to the messages parameter
# Using Union with string forward reference to avoid circular import at runtime
MessageItem = Union[Prompt, "ChatResponseContentItem", ToolCallResult]
