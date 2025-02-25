from datetime import datetime as datetime_type

from dhenara.ai.types.external_api import AIModelAPIProviderEnum, AIModelProviderEnum, ExternalApiCallStatus, ExternalApiCallStatusEnum
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseContentItem,
    ChatResponseUsage,
)
from dhenara.ai.types.shared.base import BaseModel
from django.utils import timezone


class INTStreamingProgress(BaseModel):
    """INTERNAL : Tracks the progress of a streaming response"""

    total_content: str = ""
    token_count: int = 0
    start_time: datetime_type
    last_token_time: datetime_type
    is_complete: bool = False
    model_name: str
    provider: AIModelProviderEnum
    api_provider: AIModelAPIProviderEnum


class StreamingManager:
    """Manages streaming state and constructs final ChatResponse"""

    def __init__(
        self,
        model_name: str,
        provider: AIModelProviderEnum,
        api_provider: AIModelAPIProviderEnum,
    ):
        start_time = timezone.now()
        self.metadata = {}
        self._response_metadata = {}
        self.usage: ChatResponseUsage | None = None
        self.progress = INTStreamingProgress(
            start_time=start_time,
            last_token_time=start_time,
            model_name=model_name,
            provider=provider,
            api_provider=api_provider,
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
        """Mark streaming as complete and set final usage stats"""
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

        chat_response = ChatResponse(
            model=self.progress.model_name,
            provider=self.progress.provider,
            api_provider=self.progress.api_provider,
            usage=self.usage
            or ChatResponseUsage(
                total_tokens=self.progress.token_count,
                prompt_tokens=0,  # Estimated
                completion_tokens=self.progress.token_count,
            ),
            choices=[
                ChatResponseChoice(
                    index=0,
                    content=ChatResponseContentItem(
                        role="assistant",
                        text=self.progress.total_content,
                    ),
                )
            ],
            metadata=self._response_metadata,
        )
        api_call_status = ExternalApiCallStatus(
            status=ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS,
            model=self.progress.model_name,
            api_provider=self.progress.api_provider,
            message="Streaming Completed",
            code="success",
            http_status_code=200,
        )

        return AIModelCallResponse(
            status=api_call_status,
            chat_response=chat_response,
        )
