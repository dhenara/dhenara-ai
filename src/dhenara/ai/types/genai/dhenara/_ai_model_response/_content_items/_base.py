from pydantic import Field

from dhenara.ai.types.shared.base import BaseEnum, BaseModel


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
    STRUCTURED_OUTPUT = "structured_output"
