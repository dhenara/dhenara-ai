from pydantic import Field

from dhenara.ai.types.genai.dhenara import ChatResponse, ImageResponse
from dhenara.ai.types.genai.dhenara.request import Prompt, PromptConfig, PromptMessageRoleEnum
from dhenara.ai.types.shared.base import BaseModel
from dhenara.ai.types.shared.file import GenericFile


class ConversationNode(BaseModel):
    """Represents a single turn in a conversation."""

    user_query: str
    input_files: list[GenericFile] = Field(default_factory=list)
    response: ChatResponse | ImageResponse | None = None
    timestamp: str | None = None

    def get_prompt(
        self,
        max_words_query=None,
        max_words_file=None,
    ) -> Prompt:
        return Prompt(
            role=PromptMessageRoleEnum.USER,
            text=self.user_query,
            files=self.input_files,
            config=PromptConfig(
                max_words_text=max_words_query,
                max_words_file=max_words_file,
            ),
        )

    def get_context(
        self,
        max_words_query=None,
        max_words_file=None,
        max_words_response=None,
    ) -> list[Prompt]:
        question_prompt = Prompt(
            role=PromptMessageRoleEnum.USER,
            text=self.user_query,
            files=self.input_files,
            config=PromptConfig(
                max_words_text=max_words_query,
                max_words_file=max_words_file,
            ),
        )
        response_prompt = self.response.to_prompt(
            max_words_text=max_words_response,
        )

        return [question_prompt, response_prompt]
