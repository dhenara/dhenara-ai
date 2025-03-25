from typing import Union

from pydantic import Field

from ._base import BaseResponseContentItem, ChatResponseContentItemType
from ._structured_output import ChatResponseStructuredOutput
from ._tool_call import ChatResponseToolCall


class BaseChatResponseContentItem(BaseResponseContentItem):
    type: ChatResponseContentItemType = Field(
        ...,
        description="Type of the content item",
    )
    role: str | None = Field(
        default=None,
        description="Role of the message sender in the chat context",
    )


class ChatResponseTextContentItem(BaseChatResponseContentItem):
    """Content item specific to chat responses

    Contains the role, text content, and optional function calls for chat interactions

    Attributes:
        role: The role of the message sender (system, user, assistant, or function)
        text: The actual text content of the message
        function_call: Optional function call details if the message involves function calling
    """

    type: ChatResponseContentItemType = ChatResponseContentItemType.TEXT

    text: str | None = Field(
        None,
        description="Plain text content of the message for chat interaction (without reasoning)",
    )

    def get_text(self) -> str:
        return self.text


class ChatResponseReasoningContentItem(BaseChatResponseContentItem):
    type: ChatResponseContentItemType = ChatResponseContentItemType.REASONING

    thinking_text: str | None = Field(
        None,
        description="Thinking text content, for reasoning mode",
    )

    def get_text(self) -> str:
        return self.thinking_text


class ChatResponseToolCallContentItem(BaseChatResponseContentItem):
    type: ChatResponseContentItemType = ChatResponseContentItemType.TOOL_CALL
    tool_call: ChatResponseToolCall = Field(...)

    def get_text(self) -> str:
        if self.tool_call:
            return f"Tool call: {self.tool_call.model_dump()}"
        return str(self.metadata)


class ChatResponseStructuredOutputContentItem(BaseChatResponseContentItem):
    type: ChatResponseContentItemType = ChatResponseContentItemType.STRUCTURED_OUTPUT
    structured_output: ChatResponseStructuredOutput = Field(...)

    def get_text(self) -> str:
        if self.structured_output:
            if self.structured_output.structured_data is not None:
                return f"Structured  Output: {self.structured_output.structured_data}"
            else:
                return f"Structured  Output was failed to parse. Unparsed items: {self.structured_output.model_dump()}"
        return str(self.metadata)


class ChatResponseGenericContentItem(BaseChatResponseContentItem):
    type: ChatResponseContentItemType = ChatResponseContentItemType.GENERIC
    # Use metadata to store the content

    def get_text(self) -> str:
        return str(self.metadata)


ChatResponseContentItem = Union[  # noqa: UP007
    ChatResponseTextContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseToolCallContentItem,
    ChatResponseStructuredOutputContentItem,
    ChatResponseGenericContentItem,
]


# Deltas for streamin
class BaseChatResponseContentItemDelta(BaseResponseContentItem):
    type: ChatResponseContentItemType = Field(
        ...,
        description="Type of the content item",
        serialization_alias="type",  # Ensures type is serialized correctly
    )
    role: str | None = Field(
        default=None,
        description="Role of the message sender in the chat context",
    )


class ChatResponseTextContentItemDelta(BaseChatResponseContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.TEXT

    text_delta: str | None = Field(
        None,
    )

    def get_text_delta(self) -> str:
        return self.text_delta


class ChatResponseReasoningContentItemDelta(BaseChatResponseContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.REASONING

    thinking_text_delta: str | None = Field(
        None,
    )

    def get_text_delta(self) -> str:
        return self.thinking_text_delta


# TODO: Tool call in streaming is not supported now
class ChatResponseToolCallContentItemDelta(BaseChatResponseContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.TOOL_CALL
    tool_calls_delta: str
    tool_call_deltas: list[dict] = Field(default_factory=list)

    def get_text_delta(self) -> str:
        return self.tool_calls_delta


# TODO: Structed output in streaming is not supported now
class ChatResponseStructuredOutputContentItemDelta(BaseChatResponseContentItem):
    pass  # TODO


class ChatResponseGenericContentItemDelta(BaseChatResponseContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.GENERIC
    # Use metadata to store the content

    def get_text_delta(self) -> str:
        return str(self.metadata)


ChatResponseContentItemDelta = Union[  # noqa: UP007
    ChatResponseTextContentItemDelta,
    ChatResponseReasoningContentItemDelta,
    ChatResponseToolCallContentItemDelta,
    ChatResponseStructuredOutputContentItemDelta,
    ChatResponseGenericContentItemDelta,
]
