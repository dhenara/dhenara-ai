from dhenara.ai.types.genai.ai_model import AIModelProviderEnum, ImageResponseUsage, UsageCharge
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
