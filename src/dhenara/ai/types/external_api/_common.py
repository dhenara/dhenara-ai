from pydantic import Field

from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.shared.base import BaseEnum, BaseModel


class ExternalApiCallStatusEnum(BaseEnum):
    """Status codes for AI model API endpoint responses."""

    REQUEST_SEND = "request_send"
    REQUEST_NOT_SEND = "request_not_send"
    RESPONSE_RECEIVED_SUCCESS = "response_received_success"
    RESPONSE_RECEIVED_API_ERROR = "response_received_api_error"
    RESPONSE_TIMEOUT = "response_timeout"
    INTERNAL_PROCESSING_ERROR = "internal_processing_error"


class ExternalApiCallStatus(BaseModel):
    """
    Status information for an AI model API call.

    Tracks the request/response status, provider details, and any error information
    for calls made to AI model APIs.
    """

    status: ExternalApiCallStatusEnum = Field(
        ...,
        description="Current status of the API call",
    )
    api_provider: AIModelAPIProviderEnum | str = Field(
        ...,
        description="Name of the AI API provider (e.g. 'OpenAI', 'Anthropic')",
        min_length=1,
    )
    model: str = Field(
        ...,
        description="Name of the specific AI model used",
        min_length=1,
    )
    message: str = Field(
        ...,
        description="Human readable status/error message",
    )
    code: str | None = Field(
        None,
        description="Provider-specific status/error code",
    )
    http_status_code: int | None = Field(
        None,
        description="HTTP status code if applicable",
        ge=100,
        le=599,
    )
    data: dict | None = Field(
        None,
        description="Additional provider-specific status data",
    )

    @property
    def successful(self) -> bool:
        """Check if the API call completed successfully."""
        return self.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS

    class Config:
        json_schema_extra = {
            "example": {
                "status": "response_received_success",
                "api_provider": "OpenAI",
                "model": "gpt-4",
                "message": "Request completed successfully",
                "code": "200",
                "http_status_code": 200,
                "data": {"usage": {"total_tokens": 150}},
            }
        }
