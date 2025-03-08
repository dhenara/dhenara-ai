import base64
from typing import Union

from pydantic import Field

from dhenara.ai.types.shared.base import BaseEnum, BaseModel


class ImageContentFormat(BaseEnum):
    """Enum representing different formats of image content"""

    URL = "url"
    BASE64 = "base64"
    BYTES = "bytes"
    UNKNOWN = "unknown"


class BaseResponseContentItem(BaseModel):
    """Base content item for AI model responses

    Contains common metadata fields used across different types of AI responses

    Attributes:
        metadata: System-generated metadata from API response
        storage_metadata: Storage-related metadata (e.g., cloud storage information)
        custom_metadata: User-defined additional metadata
    """

    index: int = Field(
        default=0,
        description="Content item index",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="System-generated metadata from API response processing",
    )
    storage_metadata: dict = Field(
        default_factory=dict,
        description=(
            "User-defined storage-related metadata such as cloud storage details, paths, or references. "
            "Will be empty on output from `dhenara-ai` package."
        ),
    )
    custom_metadata: dict = Field(
        default_factory=dict,
        description=(
            "User-defined additional metadata for custom processing or tracking."
            "Will be empty on output from `dhenara-ai` package"
        ),
    )


class ChatResponseContentItemType(BaseEnum):
    """Enum representing different types of content items in chat responses"""

    TEXT = "text"
    REASONING = "reasoning"
    GENERIC = "generic"
    TOOL_CALL = "tool_call"


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
    # Use metadata to store the content
    # TODO

    def get_text(self) -> str:
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


class ChatResponseToolCallContentItemDelta(BaseChatResponseContentItemDelta):
    type: ChatResponseContentItemType = ChatResponseContentItemType.TOOL_CALL
    # Use metadata to store the content
    # TODO

    def get_text_delta(self) -> str:
        return str(self.metadata)


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


class ImageResponseContentItem(BaseResponseContentItem):
    """Content item specific to image generation responses

    Contains the generated image data in various formats (bytes, base64, or URL)

    Attributes:
        content_bytes: Raw image bytes
        content_b64_json: Base64 encoded image data
        content_url: URL to the generated image
        format: Image format (e.g., PNG, JPEG)
        size: Image dimensions
    """

    content_format: ImageContentFormat = Field(
        ...,
        description="Response content format",
    )
    content_bytes: bytes | None = Field(
        None,
        description="Raw image content in bytes",
    )
    content_b64_json: str | None = Field(
        None,
        description="Base64 encoded image content",
        min_length=1,
    )
    content_url: str | None = Field(
        None,
        description="URL to access the generated image",
        pattern=r"^https?://.*$",
    )

    def validate_content(self) -> bool:
        """Validates that at least one content field is populated

        Returns:
            bool: True if at least one content field has data
        """
        return any(
            [
                self.content_bytes is not None,
                self.content_b64_json is not None,
                self.content_url is not None,
            ]
        )

    def get_content_as_bytes(self) -> bytes:
        if self.content_format == ImageContentFormat.BYTES:
            byte_content = self.content_bytes
        elif self.content_format == ImageContentFormat.BASE64:
            byte_content = base64.b64decode(self.content_b64_json)
        else:
            raise ValueError(
                f"get_content_as_bytes: Content format {self.content_format} not supported."
                "Only byte and b64_json is supported now"
            )

        return byte_content


class UsageCharge(BaseModel):
    cost: float = Field(
        ...,
        description="Cost",
    )
    charge: float | None = Field(
        ...,
        description="Charge after considering internal expences and margins."
        " Will be  None if  `cost_multiplier_percentage` is not set in cost data.",
    )
