import logging
from datetime import datetime as datetime_type

from dhenara.ai.config import settings
from dhenara.ai.types.external_api import ExternalApiCallStatus, ExternalApiCallStatusEnum
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    AIModelEndpoint,
    AIModelFunctionalTypeEnum,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseChunk,
    ChatResponseContentItemType,
    ChatResponseGenericContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCallContentItem,
    ChatResponseUsage,
    ImageResponseUsage,
    StreamingChatResponse,
    UsageCharge,
)
from dhenara.ai.types.shared.base import BaseModel

logger = logging.getLogger(__name__)


class INTStreamingProgress(BaseModel):
    """INTERNAL : Tracks the progress of a streaming response"""

    # total_content: str = ""
    updates_count: int = 0
    start_time: datetime_type
    last_token_time: datetime_type
    is_complete: bool = False
    # Add tracking for Deepseek thinking state, which is embedded in content
    in_thinking_block: bool = False


class StreamingManager:
    """Manages streaming state and constructs final ChatResponse"""

    def __init__(
        self,
        model_endpoint: AIModelEndpoint,
    ):
        self.model_endpoint = model_endpoint

        # Fields required  to create  final ChatResponse
        # self.final_response: ChatResponse | None = None
        self.usage: ChatResponseUsage | None = None
        self.usage_charge: UsageCharge | None = None
        self.choices: list[ChatResponseChoice] = []
        self.response_metadata = AIModelCallResponseMetaData(streaming=True)

        # TODO: cleanup naming
        self.provider_metadata = {}
        self.message_metadata = {}  # Anthropic
        self.persistant_choice_metadata_list = []  # OpenAI

        start_time = datetime_type.now()
        # TODO_FUTURE: Create progress per choices ?
        self.progress = INTStreamingProgress(
            start_time=start_time,
            last_token_time=start_time,
        )

    def update_usage(self, usage: ChatResponseUsage | None = None):
        """Update usgae"""
        if usage:
            self.usage = usage

    def complete(self) -> AIModelCallResponse:
        """Mark streaming as complete and set final stats"""
        self.progress.is_complete = True

        # Calculate duration
        duration = self.progress.last_token_time - self.progress.start_time
        duration_seconds = duration.total_seconds()

        self.response_metadata.duration_seconds = duration_seconds
        self.response_metadata.provider_metadata = self.provider_metadata

        return self.get_final_response()

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
                choices=self.choices,
                metadata=self.response_metadata,
            )
        else:
            logger.fatal("Streaming is only supported for Chat generation models")
            return AIModelCallResponse(
                status=ExternalApiCallStatus(
                    status=ExternalApiCallStatusEnum.INTERNAL_PROCESSING_ERROR,
                    model=self.model_endpoint.ai_model.model_name,
                    api_provider=self.model_endpoint.api.provider,
                    message=(
                        f"Model {self.model_endpoint.ai_model.model_name} not supported for streaming. "
                        "Only Chat models are supported."
                    ),
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

    def get_streaming_done_chunk(self):
        return StreamingChatResponse(
            id=None,
            data=ChatResponseChunk(
                model=self.model_endpoint.ai_model.model_name,
                provider=self.model_endpoint.ai_model.provider,
                api_provider=self.model_endpoint.api.provider,
                done=True,
            ),
        )

    def get_streaming_usage_and_charge(
        self,
    ) -> tuple[
        ChatResponseUsage | ImageResponseUsage | None,
        UsageCharge | None,
    ]:
        """Parse the OpenAI response into our standard format"""
        usage_charge = None

        if settings.ENABLE_USAGE_TRACKING or settings.ENABLE_COST_TRACKING:
            if not self.usage:
                logger.fatal("Usage not set before completing streaming.")
                return (None, None)

            if settings.ENABLE_COST_TRACKING:
                usage_charge = self.model_endpoint.calculate_usage_charge(self.usage)

        return (self.usage, usage_charge)

    def update(
        self,
        choice_deltas: list[ChatResponseChoiceDelta],
        response_metadata: dict | None = None,
    ) -> ChatResponseChunk:
        """Update streaming progress with new chunk of deltas"""
        # Update metadata if provided
        if response_metadata:
            self.response_metadata.update(response_metadata)

        # Update last token time
        self.progress.last_token_time = datetime_type.now()

        if settings.ENABLE_STREAMING_CONSOLIDATION and choice_deltas:
            # Initialize choices list if empty
            if not self.choices:
                self.choices = [ChatResponseChoice(index=i, contents=[]) for i in range(len(choice_deltas))]

            # Process each choice delta
            for choice_delta in choice_deltas:
                choice_index = choice_delta.index

                # Ensure we have enough choices initialized
                while len(self.choices) <= choice_index:
                    self.choices.append(ChatResponseChoice(index=len(self.choices), contents=[]))

                choice = self.choices[choice_index]

                # Update choice metadata
                if choice_delta.finish_reason is not None:
                    choice.finish_reason = choice_delta.finish_reason

                if choice_delta.metadata:
                    choice.metadata.update(choice_delta.metadata)

                # Process content deltas if any
                if choice_delta.content_deltas:
                    # Initialize contents list if empty
                    if not choice.contents:
                        choice.contents = []

                    for content_delta in choice_delta.content_deltas:
                        # Find matching content by type and index, or create new
                        matching_content = None

                        # First try to find exact match by type and index
                        for content in choice.contents:
                            if content and content.type == content_delta.type and content.index == content_delta.index:
                                matching_content = content
                                break

                        # If no exact match, try to find by type only
                        if not matching_content:
                            reversed_contents = list(reversed(choice.contents))
                            for content in reversed_contents:
                                if content and content.type == content_delta.type:
                                    matching_content = content
                                    break

                        # If still no match, create new content item
                        if not matching_content:
                            # Create new content based on delta type
                            if content_delta.type == ChatResponseContentItemType.TEXT:
                                matching_content = ChatResponseTextContentItem(
                                    index=content_delta.index,
                                    type=ChatResponseContentItemType.TEXT,
                                    role=content_delta.role,
                                    text="",
                                    metadata=content_delta.metadata,
                                    storage_metadata=content_delta.storage_metadata,
                                    custom_metadata=content_delta.custom_metadata,
                                )
                            elif content_delta.type == ChatResponseContentItemType.REASONING:
                                matching_content = ChatResponseReasoningContentItem(
                                    index=content_delta.index,
                                    type=ChatResponseContentItemType.REASONING,
                                    role=content_delta.role,
                                    thinking_text="",
                                    metadata=content_delta.metadata,
                                    storage_metadata=content_delta.storage_metadata,
                                    custom_metadata=content_delta.custom_metadata,
                                )
                            elif content_delta.type == ChatResponseContentItemType.TOOL_CALL:
                                matching_content = ChatResponseToolCallContentItem(
                                    index=content_delta.index,
                                    type=ChatResponseContentItemType.TOOL_CALL,
                                    role=content_delta.role,
                                    metadata=content_delta.metadata,
                                    storage_metadata=content_delta.storage_metadata,
                                    custom_metadata=content_delta.custom_metadata,
                                )
                            elif content_delta.type == ChatResponseContentItemType.GENERIC:
                                matching_content = ChatResponseGenericContentItem(
                                    index=content_delta.index,
                                    type=ChatResponseContentItemType.GENERIC,
                                    role=content_delta.role,
                                    metadata=content_delta.metadata,
                                    storage_metadata=content_delta.storage_metadata,
                                    custom_metadata=content_delta.custom_metadata,
                                )
                            else:
                                logger.error(f"stream_manager: Unknown content_delta type {content_delta.type}")
                                continue

                            choice.contents.append(matching_content)

                        # Verify type matches
                        if matching_content.type != content_delta.type or matching_content.index != content_delta.index:
                            logger.error(f"stream_manager: Content type mismatch at index {content_delta.index}")
                            continue

                        # Update content based on delta type
                        if content_delta.type == ChatResponseContentItemType.TEXT:
                            delta_text = content_delta.get_text_delta()
                            if delta_text:
                                matching_content.text = (matching_content.text or "") + delta_text

                        elif content_delta.type == ChatResponseContentItemType.REASONING:
                            delta_text = content_delta.get_text_delta()
                            if delta_text:
                                matching_content.thinking_text = (matching_content.thinking_text or "") + delta_text

                        elif content_delta.type in (
                            ChatResponseContentItemType.TOOL_CALL,
                            ChatResponseContentItemType.GENERIC,
                        ):
                            # Update metadata for tool calls and generic content
                            matching_content.metadata.update(content_delta.metadata)

        # Update token count
        self.progress.updates_count += 1

        # Create and return stream chunk
        return ChatResponseChunk(
            model=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            usage=self.usage,
            usage_charge=self.usage_charge,
            choice_deltas=choice_deltas,
            metadata=self.response_metadata,
            done=False,
        )
