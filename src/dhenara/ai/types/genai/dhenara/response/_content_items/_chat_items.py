from typing import Union

from pydantic import Field

from dhenara.ai.types.shared.base.base import BaseModel

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


class ChatMessageContentPart(BaseModel):
    """Provider-agnostic message content part.

    Designed to round-trip provider-specific content arrays (e.g., OpenAI Responses API
    output message parts like {type: "output_text", text: "...", annotations: [...]})
    while offering typed access in Dhenara models.

    We allow extra fields for forward compatibility (providers may add more keys).
    """

    type: str = Field(..., description="Content part type (e.g., output_text, input_image)")
    text: str | None = Field(default=None, description="Primary text content for text-like parts (e.g., output_text)")
    annotations: list[dict] | None = Field(
        default=None, description="Optional annotations metadata as provided by the provider"
    )
    metadata: dict | None = Field(default=None, description="Optional metadata as provided by the provider")

    # Allow unknown provider-specific fields
    model_config = {
        **BaseModel.model_config,
        "extra": "allow",
    }


class ChatResponseTextContentItem(BaseChatResponseContentItem):
    """Content item specific to chat responses

    Contains the role, text content, and optional function calls for chat interactions

    Attributes:
        role: The role of the message sender (system, user, assistant, or function)
        text: The actual text content of the message
        function_call: Optional function call details if the message involves function calling
        message_id: Provider-specific message ID (for OpenAI Responses API round-tripping)
        message_contents: Provider-specific full content array (for OpenAI Responses API round-tripping)
    """

    type: ChatResponseContentItemType = ChatResponseContentItemType.TEXT

    text: str | None = Field(
        None,
        description="Plain text content of the message for chat interaction (without reasoning)",
    )

    # Provider-specific fields for round-tripping (e.g., OpenAI Responses API)
    message_id: str | None = Field(
        None,
        description="Provider-specific message ID for round-tripping",
    )
    message_contents: list[ChatMessageContentPart] | None = Field(
        None,
        description="Provider-specific full content array for round-tripping (e.g., OpenAI output_text items)",
    )

    def get_text(self) -> str:
        if self.text:
            return self.text
        if self.message_contents:
            # Prefer concatenating text fields from output_text parts
            texts = [p.text for p in self.message_contents if getattr(p, "type", None) == "output_text" and p.text]
            if texts:
                return "".join(texts)
            return str([p.model_dump() for p in self.message_contents])
        return ""


# NOTE: LLMs outs structured as pure text with all text properties, we parse them as strucuted output with validation.
# THus structured output content items are extended from text items.
class ChatResponseStructuredOutputContentItem(ChatResponseTextContentItem):
    type: ChatResponseContentItemType = ChatResponseContentItemType.STRUCTURED_OUTPUT
    structured_output: ChatResponseStructuredOutput = Field(...)

    def get_text(self) -> str:
        if self.structured_output:
            if self.structured_output.structured_data is not None:
                return f"Structured  Output: {self.structured_output.structured_data}"
            else:
                return f"Structured  Output was failed to parse. Unparsed items: {self.structured_output.model_dump()}"
        return str(self.metadata)


class ChatResponseReasoningContentItem(BaseChatResponseContentItem):
    type: ChatResponseContentItemType = ChatResponseContentItemType.REASONING

    thinking_text: str | None = Field(
        None,
        description="Thinking text content, for reasoning mode",
    )
    thinking_id: str | None = None
    thinking_summary: str | list[ChatMessageContentPart] | None = None
    thinking_signature: str | None = None
    thinking_status: str | None = None  # OpenAI provides status as in_progress, completed, or incomplete
    metadata: dict | None = None

    def get_text(self) -> str:
        return self.thinking_text or self.thinking_summary or None


class ChatResponseToolCallContentItem(BaseChatResponseContentItem):
    type: ChatResponseContentItemType = ChatResponseContentItemType.TOOL_CALL
    tool_call: ChatResponseToolCall = Field(...)

    def get_text(self) -> str:
        if self.tool_call:
            return f"Tool call: {self.tool_call.model_dump()}"
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
    # Provider-specific fields for round-tripping (e.g., OpenAI Responses API)
    message_id: str | None = Field(
        None,
        description="Provider-specific message ID for round-tripping",
    )
    message_contents: list[ChatMessageContentPart] | None = Field(
        None,
        description="Provider-specific full content array for round-tripping (e.g., OpenAI output_text items)",
    )

    def get_text_delta(self) -> str:
        return self.text_delta


class ChatResponseReasoningContentItemDelta(BaseChatResponseContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.REASONING

    thinking_text_delta: str | None = None
    thinking_summary_delta: str | None = None  # Some models may provide a summary
    thinking_id: str | None = None
    thinking_signature: str | None = None

    def get_text_delta(self) -> str:
        return self.thinking_text_delta or self.thinking_summary_delta or None


# Tool call streaming: Providers may emit incremental tool arguments deltas and/or
# finalized tool call objects. This delta type carries either a partial arguments
# string (arguments_delta) that the StreamingManager buffers and parses on finalize,
# or a full tool_call when the provider sends a completed call.
class ChatResponseToolCallContentItemDelta(BaseChatResponseContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.TOOL_CALL
    # Optional fully-formed tool call (eg. on completed event)
    tool_call: ChatResponseToolCall | None = None
    # Optional incremental arguments delta (plain text JSON chunk)
    arguments_delta: str | None = None
    # Backward-compatible fields (not used but kept to avoid breaking callers)
    tool_calls_delta: str | None = None
    tool_call_deltas: list[dict] = Field(default_factory=list)

    def get_text_delta(self) -> str:
        # Prefer new field, fallback to legacy name if present
        return self.arguments_delta or self.tool_calls_delta


# INFO: There is no separate `structured_output` in streaming, its simply the outout text
# Structured output is derived interally from text deltas


class ChatResponseGenericContentItemDelta(BaseChatResponseContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.GENERIC
    # Use metadata to store the content

    def get_text_delta(self) -> str:
        return str(self.metadata)


ChatResponseContentItemDelta = Union[  # noqa: UP007
    ChatResponseTextContentItemDelta,
    ChatResponseReasoningContentItemDelta,
    ChatResponseToolCallContentItemDelta,
    ChatResponseGenericContentItemDelta,
]
