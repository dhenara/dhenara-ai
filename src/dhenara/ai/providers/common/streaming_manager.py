import logging
from datetime import datetime as datetime_type
from typing import Union

from dhenara.ai.config import settings
from dhenara.ai.types.external_api import ExternalApiCallStatus, ExternalApiCallStatusEnum
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    AIModelEndpoint,
    AIModelFunctionalTypeEnum,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseTextContentItem,
    ChatResponseUsage,
    ImageResponseUsage,
    UsageCharge,
)
from dhenara.ai.types.shared.base import BaseModel
from django.utils import timezone

logger = logging.getLogger(__name__)


class INTStreamingProgress(BaseModel):
    """INTERNAL : Tracks the progress of a streaming response"""

    total_content: str = ""
    token_count: int = 0
    start_time: datetime_type
    last_token_time: datetime_type
    is_complete: bool = False


class StreamingManager:
    """Manages streaming state and constructs final ChatResponse"""

    def __init__(
        self,
        model_endpoint: AIModelEndpoint,
    ):
        self.model_endpoint = model_endpoint
        self.metadata = {}
        self._response_metadata = {}
        start_time = timezone.now()
        self.usage: ChatResponseUsage | None = None
        self.progress = INTStreamingProgress(
            start_time=start_time,
            last_token_time=start_time,
        )

    def update(self, chunk: str):
        """Update streaming progress with new chunk"""

        self.progress.total_content += chunk
        self.progress.token_count += 1
        self.progress.last_token_time = timezone.now()

    def update_usage(self, usage: ChatResponseUsage | None = None):
        """Update usgae"""
        if usage:
            self.usage = usage

    def complete(self, metadata: dict | None = None):
        """Mark streaming as complete and set final stats"""
        if metadata is None:
            metadata = {}
        self.progress.is_complete = True

        # Calculate duration
        duration = self.progress.last_token_time - self.progress.start_time
        duration_seconds = duration.total_seconds()

        metadata_copy = metadata.copy()  # Create a copy of the metadata
        self.metadata.update(metadata_copy)

        _resp_data = {
            "streaming": True,
            "duration_seconds": duration_seconds,
            "token_count": self.progress.token_count,
            "provider_data": self.metadata,
        }

        try:
            self._response_metadata = AIModelCallResponseMetaData(**_resp_data)
        except:
            self._response_metadata = _resp_data

    def get_final_response(self) -> AIModelCallResponse:
        """Convert streaiming progress to ChatResponse format"""

        chat_response = None

        usage, usage_charge = self.get_streaming_usage_and_charge()

        if self.model_endpoint.ai_model.functional_type == AIModelFunctionalTypeEnum.TEXT_GENERATION:
            chat_response = ChatResponse(
                model=self.model_endpoint.ai_model.model_name,
                provider=self.model_endpoint.ai_model.provider,
                api_provider=self.model_endpoint.api.provider,
                usage=usage,
                usage_charge=usage_charge,
                choices=[
                    ChatResponseChoice(
                        index=0,
                        contents=[
                            ChatResponseTextContentItem(
                                role="assistant",
                                text=self.progress.total_content,
                            )
                        ],
                    )
                ],
                metadata=self._response_metadata,
            )
        else:
            logger.fatal("Streaming is only supported for Chat generation models")
            return AIModelCallResponse(
                status=ExternalApiCallStatus(
                    status=ExternalApiCallStatusEnum.INTERNAL_PROCESSING_ERROR,
                    model=self.model_endpoint.ai_model.model_name,
                    api_provider=self.model_endpoint.api.provider,
                    message=f"Model {self.model_endpoint.ai_model.model_name} not supported for streaming. Only Chat models are supported.",
                    code="error",
                    http_status_code=400,
                ),
            )

        api_call_status = ExternalApiCallStatus(
            status=ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS,
            model=self.model_endpoint.ai_model.model_name,
            api_provider=self.model_endpoint.api.provider,
            message="Streaming Completed",
            code="success",
            http_status_code=200,
        )

        return AIModelCallResponse(
            status=api_call_status,
            chat_response=chat_response,
            image_response=None,
        )

    def get_streaming_usage_and_charge(
        self,
    ) -> tuple[Union[ChatResponseUsage, ImageResponseUsage, None], Union[UsageCharge | None]]:
        """Parse the OpenAI response into our standard format"""
        usage_charge = None

        if settings.ENABLE_USAGE_TRACKING or settings.ENABLE_COST_TRACKING:
            if not self.usage:
                logger.fatal("Usage not set before completing streaming.")
                return (None, None)

            if settings.ENABLE_COST_TRACKING:
                usage_charge = self.model_endpoint.calculate_usage_charge(self.usage)

        return (self.usage, usage_charge)
