from typing import Any

from pydantic import Field

from dhenara.ai.types.genai.ai_model import (
    AIModelAPIProviderEnum,
    AIModelProviderEnum,
    ChatResponseUsage,
    UsageCharge,
)
from dhenara.ai.types.genai.dhenara.request import PromptMessageRoleEnum
from dhenara.ai.types.genai.dhenara.request.data import Content, Prompt, PromptText
from dhenara.ai.types.shared.api import SSEEventType, SSEResponse
from dhenara.ai.types.shared.base import BaseModel

from ._content_items._chat_items import ChatResponseContentItem, ChatResponseContentItemDelta
from ._metadata import AIModelCallResponseMetaData


class ChatResponseChoice(BaseModel):
    """A single choice/completion in the chat response"""

    index: int
    finish_reason: Any | None = None
    stop_sequence: Any | None = None
    contents: list[ChatResponseContentItem] | None = None
    metadata: dict = {}

    class Config:
        json_schema_extra = {
            "example": {
                "index": 0,
                "contents": [
                    {
                        "role": "assistant",
                        "text": "Hello! How can I help you today?",
                    }
                ],
            }
        }


class ChatResponseChoiceDelta(BaseModel):
    """A single choice/completion in the chat response"""

    index: int
    finish_reason: Any | None = None
    stop_sequence: Any | None = None
    content_deltas: list[ChatResponseContentItemDelta] | None = None
    metadata: dict = {}


class ChatResponse(BaseModel):
    """Complete chat response from an AI model

    Contains the response content, usage statistics, and provider-specific metadata
    """

    model: str
    provider: AIModelProviderEnum
    api_provider: AIModelAPIProviderEnum | None = None
    usage: ChatResponseUsage | None = None
    usage_charge: UsageCharge | None = None
    choices: list[ChatResponseChoice] = []
    metadata: AIModelCallResponseMetaData | dict = {}

    def get_visible_fields(self) -> dict:
        return self.model_dump(exclude=["choices"])

    # Add to ChatResponse class in _chat.py
    def to_prompt(self) -> "Prompt":
        """Convert response to a context message for next turn"""

        # Get text from the first choice's contents
        if not self.choices:
            return None

        choice = self.choices[0]
        if not choice.contents:
            return None

        # Combine all content items into one text
        text_parts = [content_item.get_text() for content_item in choice.contents]

        text = "\n".join(text_parts)

        # Create Content object
        content = Content(type="text", text=text)

        # Create PromptText object
        prompt_text = PromptText(content=content)

        # Create and return Prompt object
        return Prompt(role=PromptMessageRoleEnum.ASSISTANT, text=prompt_text)


class ChatResponseChunk(BaseModel):
    """Chat response Chunk from an AI model

    Contains the response content, usage statistics, and provider-specific metadata
    """

    model: str
    provider: AIModelProviderEnum
    api_provider: AIModelAPIProviderEnum | None = None
    usage: ChatResponseUsage | None = None
    usage_charge: UsageCharge | None = None
    choice_deltas: list[ChatResponseChoiceDelta] = []
    metadata: AIModelCallResponseMetaData | dict = {}

    done: bool = Field(
        default=False,
        description="Indicates if this is the final chunk",
    )

    def get_visible_fields(self) -> dict:
        return self.model_dump(exclude=["choices"])


class StreamingChatResponse(SSEResponse[ChatResponseChunk]):
    """Specialized SSE response for chat streaming"""

    event: SSEEventType = SSEEventType.TOKEN_STREAM
    data: ChatResponseChunk
