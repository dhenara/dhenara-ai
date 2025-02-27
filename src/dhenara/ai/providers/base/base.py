import logging
from abc import ABC, abstractmethod
from typing import Any, Union

from dhenara.ai.config import settings
from dhenara.ai.types import (
    AIModelCallConfig,
    AIModelCallResponse,
    AIModelEndpoint,
    AIModelFunctionalTypeEnum,
    ChatResponseGenericContentItem,
    ChatResponseGenericContentItemDelta,
    ChatResponseUsage,
    ImageResponseUsage,
    UsageCharge,
)
from dhenara.ai.types.external_api import ExternalApiCallStatus, ExternalApiCallStatusEnum, FormattedPrompt, SystemInstructions
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse

logger = logging.getLogger(__name__)


class AIModelProviderClientBase(ABC):
    """Base class for AI model provider handlers"""

    prompt_message_class = None

    def __init__(self, model_endpoint: AIModelEndpoint, config: AIModelCallConfig):
        self.model_endpoint = model_endpoint
        self.model_name_in_api_calls = self.model_endpoint.ai_model.model_name_with_version_suffix
        self.config = config
        self._client = None
        self._initialized = False
        self._input_validation_pending = True

    async def __aenter__(self):
        if not self._initialized:
            self._client = await self._setup_client()
            await self.initialize()
            self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        # INFO: Don't close the client here, it will be managed by the provider client
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass

    @abstractmethod
    def process_instructions(
        self,
        instructions: SystemInstructions,
    ) -> FormattedPrompt | str | None:
        pass

    @abstractmethod
    async def generate_response(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: SystemInstructions | None = None,
    ) -> AIModelCallResponse:
        """Generate response from the model"""
        pass

    async def _validate_and_generate_response(
        self,
        prompt: FormattedPrompt,
        context: list[FormattedPrompt] | None = None,
        instructions: SystemInstructions | None = None,
    ) -> AIModelCallResponse:
        """Generate response from the model"""

        if settings.ENABLE_PROMPT_VALIDATION:
            validated_inputs = self.validate_inputs(prompt=prompt, context=context)
            if not validated_inputs:
                return AIModelCallResponse(
                    status=self._create_error_status(
                        message="Input validation failed, not proceeding to generation.",
                        status=ExternalApiCallStatusEnum.REQUEST_NOT_SEND,
                    )
                )
            return await self.generate_response(
                prompt=validated_inputs[0],
                context=validated_inputs[1],
                instructions=instructions,
            )
        else:
            self._input_validation_pending = False
            return await self.generate_response(
                prompt=prompt,
                context=context,
                instructions=instructions,
            )

    def validate_inputs(
        self,
        prompt: FormattedPrompt | dict,
        context: list[FormattedPrompt | dict] | None = None,
    ) -> tuple[FormattedPrompt, list[FormattedPrompt] | None] | None:
        try:
            validated_prompt = self._validate_prompt(prompt)
            validated_context = [self._validate_prompt(pmt) for pmt in context]

            if not self.validate_options():
                logger.debug("validate_inputs: ERROR: validate_options failed")
                return None

            self._input_validation_pending = False
            return validated_prompt, validated_context
        except Exception as e:
            logger.exception(f"validate_inputs: {e}")
            return None

    def _validate_prompt(
        self,
        prompt: FormattedPrompt | dict,
    ) -> tuple[dict, list[dict] | None]:
        if isinstance(prompt, self.prompt_message_class):
            validated = prompt
        elif isinstance(prompt, dict):
            validated = self.prompt_message_class(**prompt)
        elif isinstance(prompt, str) and self.model_endpoint.ai_model.functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
            validated = prompt
            return prompt
        else:
            raise ValueError(f"Prompt type {type(prompt)} not valid")

        return validated.model_dump()

    def validate_options(self) -> bool:
        """Validate configuration options"""
        model_options = self.config.options
        return self.model_endpoint.ai_model.validate_options(model_options)

    # -------------------------------------------------------------------------
    # For Usage and cost
    @abstractmethod
    def _get_usage_from_provider_response(self, response):
        pass

    def get_usage_and_charge(
        self,
        response,
        usage=None,
    ) -> tuple[Union[ChatResponseUsage, ImageResponseUsage, None], Union[UsageCharge | None]]:
        """Parse the OpenAI response into our standard format"""
        usage = None
        usage_charge = None

        if settings.ENABLE_USAGE_TRACKING or settings.ENABLE_COST_TRACKING:
            if usage is None:
                usage = self._get_usage_from_provider_response(response)

            if settings.ENABLE_COST_TRACKING:
                usage_charge = self.model_endpoint.calculate_usage_charge(usage)

        return (usage, usage_charge)

    # -------------------------------------------------------------------------
    def _create_success_status(
        self,
        message: str = "Output generated",
        status: ExternalApiCallStatusEnum = ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS,
    ) -> ExternalApiCallStatus:
        """Create success status response"""
        return ExternalApiCallStatus(
            status=status,
            api_provider=self.model_endpoint.api.provider,
            model=self.model_endpoint.ai_model.model_name,
            message=message,
            code="success",
            http_status_code=200,
        )

    def _create_error_status(
        self,
        message: str,
        status: ExternalApiCallStatusEnum = ExternalApiCallStatusEnum.REQUEST_NOT_SEND,
    ) -> ExternalApiCallStatus:
        """Create error status response"""
        return ExternalApiCallStatus(
            status=status,
            api_provider=self.model_endpoint.api.provider,
            model=self.model_endpoint.ai_model.model_name,
            message=message,
            code="external_api_error",
            http_status_code=400,
        )

    def _create_streaming_error_response(self, exc: Exception | None = None, message: str | None = None):
        if exc:
            logger.exception(f"Error during streaming: {exc}")

        if message:
            detail_msg = message
        elif exc:
            detail_msg = f"Error: {exc}"
        else:
            detail_msg = "Streaming Error"

        return SSEErrorResponse(
            data=SSEErrorData(
                error_code=SSEErrorCode.external_api_error,
                message=f"Error While Streaming: {detail_msg}",
                details={
                    "error": detail_msg,
                },
            )
        )

    def get_unknown_content_type_item(self, role: str, unknown_item: Any, streaming: bool):
        logger.debug(f"process_content_item_delta: Unknown content item type {type(unknown_item)}")
        try:
            data = unknown_item.model_dump()
        except:
            data = None

        item_dict = {
            "role": role,
            "metadata": {
                "unknonwn": f"Unknown content item of type {type(unknown_item)}",
                "data": data,
            },
        }
        return ChatResponseGenericContentItemDelta(**item_dict) if streaming else ChatResponseGenericContentItem(**item_dict)
