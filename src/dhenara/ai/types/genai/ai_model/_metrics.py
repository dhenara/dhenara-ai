from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel


class UsageCharge(BaseModel):
    cost: float = Field(
        ...,
        description="Cost",
    )
    charge: float | None = Field(
        default=None,
        description="Charge after considering internal expences and margins."
        " Will be  None if  `cost_multiplier_percentage` is not set in cost data.",
    )


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
