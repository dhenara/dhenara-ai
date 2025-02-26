from typing import Any

from dhenara.ai.types.external_api._providers import AIModelAPIProviderEnum, AIModelProviderEnum
from dhenara.ai.types.shared.api import SSEDataChunk, SSEEventType, SSEResponse
from dhenara.ai.types.shared.base import BaseModel

from ._content_item import ChatResponseContentItem, UsageCharge


class AIModelCallResponseMetaData(BaseModel):
    streaming: bool = False
    duration_seconds: int | float | None = None
    provider_data: dict


class ChatResponseChoice(BaseModel):
    """A single choice/completion in the chat response"""

    index: int
    finish_reason: Any | None = None
    stop_sequence: Any | None = None
    contents: list[ChatResponseContentItem] | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "index": 0,
                "content": {
                    "role": "assistant",
                    "text": "Hello! How can I help you today?",
                },
            }
        }


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


class ChatResponseMetaDataOpenAi(BaseModel):
    """OpenAI specific metadata for chat responses"""

    id: str
    object: str  # object type : chat.completion
    created: int | None = None  # Unix timestamp (in seconds) of creation
    system_fingerprint: str


class ChatResponseMetaDataGoogleAi(BaseModel):
    """Google AI specific metadata for chat responses"""

    prompt_feedback: dict | None = None


class ChatResponseMetaDataAnthropic(BaseModel):
    """Anthropic specific metadata for chat responses"""

    id: str
    type: str


# Streaming
class TokenStreamChunk(SSEDataChunk):
    """Specialized chunk for token streaming"""

    pass
    # token_count: int | None = Field(
    #    default=None,
    #    description="Number of tokens in this chunk",
    # )
    # model: str | None = Field(
    #    default=None,
    #    description="Model generating the tokens",
    # )


class StreamingChatResponse(SSEResponse[TokenStreamChunk]):
    """Specialized SSE response for chat streaming"""

    event: SSEEventType = SSEEventType.TOKEN_STREAM
    data: TokenStreamChunk


class ChatResponse(BaseModel):
    """Complete chat response from an AI model

    Contains the response content, usage statistics, and provider-specific metadata
    """

    model: str
    provider: AIModelProviderEnum
    api_provider: AIModelAPIProviderEnum | None = None
    usage: ChatResponseUsage | None
    usage_charge: UsageCharge | None
    choices: list[ChatResponseChoice]
    metadata: AIModelCallResponseMetaData | dict = {}

    def get_visible_fields(self) -> dict:
        return self.model_dump(exclude=["choices"])
