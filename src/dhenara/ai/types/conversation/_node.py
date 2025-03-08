
from pydantic import Field

from dhenara.ai.types.genai.dhenara import ChatResponse, ImageResponse
from dhenara.ai.types.shared.base import BaseModel
from dhenara.ai.types.shared.file import GenericFile


class ConversationNode(BaseModel):
    """Represents a single turn in a conversation."""
    user_query: str
    attached_files: list[GenericFile] = Field(default_factory=list)
    response: ChatResponse | ImageResponse | None = None
    timestamp: str | None = None
