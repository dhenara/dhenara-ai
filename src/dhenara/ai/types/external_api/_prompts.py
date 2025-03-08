import logging
from typing import Annotated, Any, Literal, Union

from pydantic import Field, field_validator

from dhenara.ai.types.external_api import (
    AnthropicMessageRoleEnum,
    GoogleAiMessageRoleEnum,
    OpenAiMessageRoleEnum,
)
from dhenara.ai.types.shared.base import BaseModel

logger = logging.getLogger(__name__)


# OpenAI Specific Models


class OpenAITextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class OpenAIImageUrlContent(BaseModel):
    url: str


class OpenAIImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: OpenAIImageUrlContent


# Discriminated union for content parts
ContentPart = Annotated[
    OpenAITextContent | OpenAIImageContent,
    Field(discriminator="type"),
]


class OpenAIPromptMessage(BaseModel):
    role: OpenAiMessageRoleEnum
    content: str | list[ContentPart]
    function_call: dict[str, Any] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v):
        # Handle string case
        if isinstance(v, str):
            return v

        # Handle list case with automatic discrimination
        if isinstance(v, list):
            return [{"type": item.get("type"), **item} for item in v]

        raise ValueError("Content must be string or list of content parts")

    ## fmt: off
    # def model_dump(self, **kwargs):
    #    # Clean None values and empty lists
    #    data = super().model_dump(**kwargs)

    #    if isinstance(data["content"], list):
    #        data["content"] = [
    #            part for part in data["content"]
    #            if (part["type"] == "text" and part["text"]) or
    #               (part["type"] == "image_url" and part["image_url"]["url"])
    #            ]

    #    if not data["content"] and isinstance(data["content"], list):
    #        data["content"] = ""  # Fallback to empty string

    #    return data


# Anthropic Specific Models


class AnthropicMessageTextContent(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicMessageImageContent(BaseModel):
    type: Literal["image"]
    source: dict  # Or use a more specific type for image source


class AnthropicPromptMessage(BaseModel):
    role: AnthropicMessageRoleEnum
    content: str | list[AnthropicMessageTextContent | AnthropicMessageImageContent]

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if isinstance(v, str):
            return v

        for item in v:
            if isinstance(item, dict):
                if item["type"] == "text":
                    if "source" in item:
                        item.pop("source")  # Remove source field for text content
                elif item["type"] == "image" and "source" not in item:
                    raise ValueError("source is required for image type")
        return v


# Google AI Specific Models


class GoogleAIInlineData(BaseModel):
    """Model for inline data like images"""

    data: str
    mime_type: str


class GoogleAIPart(BaseModel):
    """Base model for different types of content parts"""

    text: str | None = None
    inline_data: GoogleAIInlineData | None = None

    @field_validator("inline_data", mode="before")
    @classmethod
    def validate_inline_data(cls, v: Any) -> GoogleAIInlineData | None:
        if v is None:
            return None
        if isinstance(v, dict):
            return GoogleAIInlineData(**v)
        return v

    # def model_dump(self) -> dict[str, Any]:
    #    if self.text is not None:
    #        return {"text": self.text}
    #    if self.inline_data is not None:
    #        return {"inline_data": self.inline_data.model_dump()}
    #    return {}


class GoogleAIPromptMessage(BaseModel):
    """A complete Google AI prompt message"""

    role: GoogleAiMessageRoleEnum
    parts: list[GoogleAIPart]

    @field_validator("parts", mode="before")
    @classmethod
    def validate_parts(cls, v: Any) -> list[GoogleAIPart]:
        if isinstance(v, str):
            return [GoogleAIPart(text=v)]

        if isinstance(v, list):
            validated_parts = []
            for part in v:
                if isinstance(part, str):
                    validated_parts.append(GoogleAIPart(text=part))
                elif isinstance(part, dict):
                    validated_parts.append(GoogleAIPart(**part))
                elif isinstance(part, GoogleAIPart):
                    validated_parts.append(part)
                else:
                    raise ValueError(f"Invalid part type: {type(part)}")
            return validated_parts

        raise ValueError(f"Invalid parts type: {type(v)}")

    # def model_dump(self) -> dict[str, Any]:
    #    return {"role": self.role.value, "parts": [part.model_dump() for part in self.parts]}


# # Configuration Models
# class PromptConfig(BaseModel):
#     model_provider: AIModelProviderEnum
#     max_tokens_query: int | None = Field(default=None, gt=0)
#     max_tokens_files: int | None = Field(default=None, gt=0)
#     max_tokens_response: int | None = Field(default=None, gt=0)
#     response_before_query: bool = False
#
#
# class ConversationNodeContent(BaseModel):
#     user_query: str | None = None
#     attached_files: list[GenericFile] | None = None
#     previous_response: Union[ChatResponse, ImageResponse] | None = None


FormattedPrompt = Union[OpenAIPromptMessage, GoogleAIPromptMessage, AnthropicPromptMessage]  # noqa: UP007
SystemInstructions = list[str]
