from typing import Any

from pydantic import Field

from dhenara.ai.types.external_api._providers import AIModelAPIProviderEnum, AIModelProviderEnum
from dhenara.ai.types.shared.api import SSEEventType, SSEResponse
from dhenara.ai.types.shared.base import BaseModel

from ._content_item import ChatResponseContentItem, ChatResponseContentItemDelta, UsageCharge


class AIModelCallResponseMetaData(BaseModel):
    streaming: bool = False
    duration_seconds: int | float | None = None
    provider_metadata: dict | None = None


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


class ChatResponseUsage(BaseModel):
    """Token usage statistics for the chat completion"""

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

    class Config:
        json_schema_extra = {
            "example": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            }
        }


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
