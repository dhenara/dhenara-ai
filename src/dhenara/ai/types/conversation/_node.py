from typing import List, Optional
from pydantic import Field
from dhenara.ai.types.shared.file import GenericFile
from dhenara.ai.types.genai.dhenara import ChatResponse, ImageResponse
from dhenara.ai.types.shared.base import BaseModel

class ConversationNode(BaseModel):
    """Represents a single turn in a conversation."""
    user_query: str
    attached_files: List[GenericFile] = Field(default_factory=list)
    response: Optional[ChatResponse | ImageResponse] = None
    timestamp: Optional[str] = None
