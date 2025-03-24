from pydantic import Field

from dhenara.ai.types.genai.dhenara import ChatResponse, ImageResponse
from dhenara.ai.types.genai.dhenara.request import Prompt, PromptMessageRoleEnum
from dhenara.ai.types.shared.base import BaseModel
from dhenara.ai.types.shared.file import GenericFile


class ConversationNode(BaseModel):
    """Represents a single turn in a conversation."""

    user_query: str
    attached_files: list[GenericFile] = Field(default_factory=list)
    response: ChatResponse | ImageResponse | None = None
    timestamp: str | None = None

    def get_context(self) -> list[Prompt]:
        question_prompt = Prompt(
            role=PromptMessageRoleEnum.USER,
            text=self.user_query,
        )
        response_prompt = self.response.to_prompt()

        return [question_prompt, response_prompt]
