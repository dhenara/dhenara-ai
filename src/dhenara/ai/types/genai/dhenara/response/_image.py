from dhenara.ai.types.genai.ai_model import (
    AIModelProviderEnum,
    ImageResponseUsage,
    UsageCharge,
)
from dhenara.ai.types.genai.dhenara.request import PromptMessageRoleEnum
from dhenara.ai.types.genai.dhenara.request.data import Prompt, PromptConfig
from dhenara.ai.types.shared.base import BaseModel

from ._content_items._image_items import ImageResponseContentItem
from ._metadata import AIModelCallResponseMetaData


class ImageResponseChoice(BaseModel):
    """A single image generation choice/result"""

    index: int
    contents: list[ImageResponseContentItem] = []

    class Config:
        json_schema_extra = {
            "example": {
                "index": 0,
                "content": {
                    "content_format": "url",
                    "content_url": "https://api.example.com/images/123.jpg",
                },
            }
        }


class ImageResponse(BaseModel):
    """Complete response from an AI image generation model

    Contains the generated images, usage information, and provider-specific metadata
    """

    model: str
    provider: AIModelProviderEnum
    usage: ImageResponseUsage | None
    usage_charge: UsageCharge | None
    choices: list[ImageResponseChoice]
    metadata: AIModelCallResponseMetaData | dict = {}

    def to_prompt(
        self,
        choice_index: int = 0,
        max_words_text: int | None = None,
    ) -> "Prompt":
        """Convert response to a context message for next turn"""

        # Get text from the first choice's contents
        if not self.choices:
            return None

        choice = self.choices[choice_index]
        if not choice.contents:
            return None

        # TODO
        # Process generated images if availae and add as files
        image_files = []

        # Create and return Prompt object
        return Prompt(
            role=PromptMessageRoleEnum.ASSISTANT,
            text="Generated Image",
            files=image_files,
            config=PromptConfig(
                max_words_text=max_words_text,
                max_words_file=None,
            ),
        )

    def preview_dict(self):
        """
        Returns a preview version of the response excluding the full content of choices
        but including metadata about them
        """
        _dict = self.model_dump(exclude=["choices"])

        # Add summary information about choices instead of full content
        choice_summaries = []
        for choice in self.choices:
            choice_summary = {
                "index": choice.index,
                "content_count": len(choice.contents),
                "contents_summary": [
                    {
                        "index": content.index,
                        "content_format": content.content_format,
                    }
                    for content in choice.contents
                ],
            }
            choice_summaries.append(choice_summary)

        _dict["choices_summary"] = choice_summaries
        return _dict
