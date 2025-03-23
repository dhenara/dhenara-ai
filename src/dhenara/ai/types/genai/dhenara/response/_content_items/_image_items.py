import base64

from pydantic import Field

from dhenara.ai.types.shared.base import BaseEnum, BaseModel

from ._base import BaseResponseContentItem


class ImageContentFormat(BaseEnum):
    """Enum representing different formats of image content"""

    URL = "url"
    BASE64 = "base64"
    BYTES = "bytes"
    UNKNOWN = "unknown"


class ImageResponseContentItem(BaseResponseContentItem):
    """Content item specific to image generation responses

    Contains the generated image data in various formats (bytes, base64, or URL)

    Attributes:
        content_bytes: Raw image bytes
        content_b64_json: Base64 encoded image data
        content_url: URL to the generated image
        format: Image format (e.g., PNG, JPEG)
        size: Image dimensions
    """

    content_format: ImageContentFormat = Field(
        ...,
        description="Response content format",
    )
    content_bytes: bytes | None = Field(
        None,
        description="Raw image content in bytes",
    )
    content_b64_json: str | None = Field(
        None,
        description="Base64 encoded image content",
        min_length=1,
    )
    content_url: str | None = Field(
        None,
        description="URL to access the generated image",
        pattern=r"^https?://.*$",
    )

    def validate_content(self) -> bool:
        """Validates that at least one content field is populated

        Returns:
            bool: True if at least one content field has data
        """
        return any(
            [
                self.content_bytes is not None,
                self.content_b64_json is not None,
                self.content_url is not None,
            ]
        )

    def get_content_as_bytes(self) -> bytes:
        if self.content_format == ImageContentFormat.BYTES:
            byte_content = self.content_bytes
        elif self.content_format == ImageContentFormat.BASE64:
            byte_content = base64.b64decode(self.content_b64_json)
        else:
            raise ValueError(
                f"get_content_as_bytes: Content format {self.content_format} not supported."
                "Only byte and b64_json is supported now"
            )

        return byte_content


class UsageCharge(BaseModel):
    cost: float = Field(
        ...,
        description="Cost",
    )
    charge: float | None = Field(
        ...,
        description="Charge after considering internal expences and margins."
        " Will be  None if  `cost_multiplier_percentage` is not set in cost data.",
    )
