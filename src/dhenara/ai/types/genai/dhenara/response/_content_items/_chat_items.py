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
    """Content item for assistant/user text leveraging provider `message_contents` exclusively.

    CHANGE: Removed legacy plain `text` storage. All textual content must be present inside
    `message_contents` parts (e.g., OpenAI output_text, Anthropic text, Google text parts).
    This guarantees a single source of truth for unified streaming and avoids divergence
    between incremental deltas and final aggregation.
    """

    type: ChatResponseContentItemType = ChatResponseContentItemType.TEXT

    # Provider-specific message id (e.g., OpenAI Responses API item id)
    message_id: str | None = Field(
        None,
        description="Provider-specific message/content identifier for round-tripping",
    )
    # Unified list of provider content parts (output_text, text, etc.)
    message_contents: list[ChatMessageContentPart] | None = Field(
        None,
        description=(
            "Unified provider content parts array (e.g., output_text entries). Required for text reconstruction."
        ),
    )

    def get_text(self) -> str:
        if self.message_contents:
            texts = [p.text for p in self.message_contents if getattr(p, "text", None)]
            if texts:
                return "".join(texts)
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


class ChatResponseReasoningContentItem(ChatResponseTextContentItem):
    """Reasoning content item extending text item.

    CHANGE: Removed standalone `thinking_text`; reasoning textual tokens are now
    stored inside `message_contents` parts (e.g., type="thinking" or provider-specific types).
    We retain summary/signature/id/status metadata for higher-level introspection.
    """

    type: ChatResponseContentItemType = ChatResponseContentItemType.REASONING
    thinking_id: str | None = None
    thinking_summary: list[ChatMessageContentPart] | None = None  # NOTE: Only applicable for OpenAI SDK
    thinking_signature: str | None = None
    thinking_status: str | None = None  # Provider status (in_progress, completed, etc.)
    metadata: dict | None = None

    def get_text(self) -> str:
        # Prefer reconstructed message_contents text; fallback to summary when present.
        base = super().get_text()
        if base:
            return base
        if isinstance(self.thinking_summary, list):
            parts = [p.text for p in self.thinking_summary if getattr(p, "text", None)]
            return "".join(parts)
        return ""


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

    # Unified incremental text delta (was `text_delta`); keep name for backward compat but treat as raw append
    text_delta: str | None = Field(None, description="Incremental text delta append for streaming")
    # Provider may start emitting full `message_contents` array mid-stream; we adopt it immediately.
    message_id: str | None = Field(None, description="Provider-specific message content identifier")
    message_contents: list[ChatMessageContentPart] | None = Field(
        None,
        description="Full provider content parts snapshot when available (supersedes text_delta accumulation).",
    )

    def get_text_delta(self) -> str:
        return self.text_delta or ""


class ChatResponseReasoningContentItemDelta(ChatResponseTextContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.REASONING

    # Reasoning now appends tokens via message_contents-like representation; keep delta for summaries/signature
    thinking_summary_delta: str | None = None
    thinking_id: str | None = None
    thinking_signature: str | None = None

    def get_text_delta(self) -> str:
        return self.text_delta or ""

    # NOTE: Do NOT shadow the field name with a method of the same name.
    # Historically a method named `thinking_summary_delta()` existed here,
    # which overwrote the dataclass field on instances, causing the value
    # to become a function object at runtime.
    def get_thinking_summary_delta(self) -> str:
        return self.thinking_summary_delta or ""


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
