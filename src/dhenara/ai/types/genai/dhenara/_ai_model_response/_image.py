from pydantic import Field

from dhenara.ai.types.external_api._providers import AIModelProviderEnum
from dhenara.ai.types.shared.base import BaseModel

from ._chat import AIModelCallResponseMetaData
from ._content_item import ImageResponseContentItem, UsageCharge


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


class ImageResponseUsage(BaseModel):
    """Usage information for image generation.
    Note that, for images, no usage data is received, so this class holds params required for usage/cost calculation"""

    number_of_images: int = Field(
        ...,
        description="Number of Images generated",
    )
    model: str = Field(
        default_factory=dict,
        description="Model Name",
    )
    options: dict = Field(
        default_factory=dict,
        description="Options send to API",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "dall-e-3",
                "options": {
                    "size": "1024x1024",
                    "quality": "standard",
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
