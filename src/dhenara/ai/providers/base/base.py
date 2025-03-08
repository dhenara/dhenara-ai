import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Any

from dhenara.ai.config import settings
from dhenara.ai.providers.base import StreamingManager
from dhenara.ai.types import (
    AIModelCallConfig,
    AIModelCallResponse,
    AIModelEndpoint,
    AIModelFunctionalTypeEnum,
    ChatResponse,
    ChatResponseGenericContentItem,
    ChatResponseGenericContentItemDelta,
    ChatResponseUsage,
    ImageResponse,
    ImageResponseUsage,
    StreamingChatResponse,
    UsageCharge,
)
from dhenara.ai.types.external_api import (
    ExternalApiCallStatus,
    ExternalApiCallStatusEnum,
    FormattedPrompt,
    SystemInstructions,
)
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse

logger = logging.getLogger(__name__)


class AIModelProviderClientBase(ABC):
    """Base class for AI model provider handlers"""

    prompt_message_class = None

    def __init__(self, model_endpoint: AIModelEndpoint, config: AIModelCallConfig, is_async: bool = True):
        self.model_endpoint = model_endpoint
        self.model_name_in_api_calls = self.model_endpoint.ai_model.model_name_with_version_suffix
        self.config = config
        self.is_async = is_async
        self._client = None
        self._initialized = False
        self._input_validation_pending = True
        self.streaming_manager = None

    async def __aenter__(self):
        if self.is_async:
            if not self._initialized:
                self._client = await self._setup_client_async()
                await self._initialize_async()
                self._initialized = True
            return self
        raise RuntimeError("Use 'with' for synchronous client")

    def __enter__(self):
        if not self.is_async:
            if not self._initialized:
                self._client = self._setup_client_sync()
                self._initialize_sync()
                self._initialized = True
            return self
        raise RuntimeError("Use 'async with' for asynchronous client")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.is_async:
            await self._cleanup_async()
            # INFO: Don't close the client here, it will be managed by the provider client
            self._initialized = False
            self._initialized = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_async:
            self._cleanup_sync()
            self._initialized = False

    def _initialize_sync(self) -> None:
        self.initialize()

    async def _initialize_async(self) -> None:
        self.initialize()

    def _cleanup_sync(self) -> None:
        self.cleanup()

    async def _cleanup_async(self) -> None:
        self.cleanup()

    def _setup_client_sync(self):
        if not self.is_async:
            raise NotImplementedError("_setup_client_sync")

    async def _setup_client_async(self):
        if self.is_async:
            raise NotImplementedError("_setup_client_async")

    def generate_response_sync(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: SystemInstructions | None = None,
    ) -> AIModelCallResponse:
        parsed_response: ChatResponse | None = None
        api_call_status: ExternalApiCallStatus | None = None

        logger.debug(f"generate_response: prompt={prompt}, context={context}")

        api_call_params = self.get_api_call_params(
            prompt=prompt,
            context=context,
            instructions=instructions,
        )

        logger.debug(f"generate_response: api_call_params: {api_call_params}")

        if self.config.test_mode:
            from dhenara.ai.providers.common.dummy import DummyAIModelResponseFns

            self.streaming_manager = StreamingManager(model_endpoint=self.model_endpoint)
            dummy_resp = DummyAIModelResponseFns(streaming_manager=self.streaming_manager)

            return dummy_resp.get_dummy_ai_model_response_sync(
                ai_model_ep=self.model_endpoint,
                streaming=self.config.streaming,
            )

        try:
            if self.config.streaming:
                stream = self.do_streaming_api_call_sync(api_call_params)
                stream_generator = self._handle_streaming_response_sync(stream=stream)
                return AIModelCallResponse(sync_stream_generator=stream_generator)

            response = self.do_api_call_sync(api_call_params)
            parsed_response = self.parse_response(response)
            api_call_status = self._create_success_status()
            return self._get_ai_model_call_response(parsed_response, api_call_status)

        except Exception as e:
            logger.exception(f"Error in generate_response_sync: {e}")
            api_call_status = self._create_error_status(str(e))

    async def generate_response_async(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: SystemInstructions | None = None,
    ) -> AIModelCallResponse:
        parsed_response: ChatResponse | None = None
        api_call_status: ExternalApiCallStatus | None = None

        logger.debug(f"generate_response: prompt={prompt}, context={context}")

        api_call_params = self.get_api_call_params(
            prompt=prompt,
            context=context,
            instructions=instructions,
        )

        logger.debug(f"generate_response: api_call_params: {api_call_params}")

        if self.config.test_mode:
            from dhenara.ai.providers.common.dummy import DummyAIModelResponseFns

            self.streaming_manager = StreamingManager(model_endpoint=self.model_endpoint)
            dummy_resp = DummyAIModelResponseFns(streaming_manager=self.streaming_manager)

            return await dummy_resp.get_dummy_ai_model_response_async(
                ai_model_ep=self.model_endpoint,
                streaming=self.config.streaming,
            )

        try:
            if self.config.streaming:
                stream = await self.do_streaming_api_call_async(api_call_params)
                stream_generator = self._handle_streaming_response_async(stream=stream)
                return AIModelCallResponse(async_stream_generator=stream_generator)

            response = await self.do_api_call_async(api_call_params)
            parsed_response = self.parse_response(response)
            api_call_status = self._create_success_status()
            return self._get_ai_model_call_response(parsed_response, api_call_status)

        except Exception as e:
            logger.exception(f"Error in generate_response_async: {e}")
            api_call_status = self._create_error_status(str(e))

    def _validate_and_generate_response_sync(
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
            return self.generate_response_sync(
                prompt=validated_inputs[0],
                context=validated_inputs[1],
                instructions=instructions,
            )
        else:
            self._input_validation_pending = False
            return self.generate_response_sync(
                prompt=prompt,
                context=context,
                instructions=instructions,
            )

    async def _validate_and_generate_response_async(
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
            return await self.generate_response_async(
                prompt=validated_inputs[0],
                context=validated_inputs[1],
                instructions=instructions,
            )
        else:
            self._input_validation_pending = False
            return await self.generate_response_async(
                prompt=prompt,
                context=context,
                instructions=instructions,
            )

    def _get_ai_model_call_response(self, parsed_response, api_call_status):
        functional_type = self.model_endpoint.ai_model.functional_type
        if functional_type == AIModelFunctionalTypeEnum.TEXT_GENERATION:
            return AIModelCallResponse(
                status=api_call_status,
                chat_response=parsed_response,
            )
        elif functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
            return AIModelCallResponse(
                status=api_call_status,
                image_response=parsed_response,
            )
        else:
            raise ValueError(f"_get_ai_model_call_response: Unknown functional_type {functional_type}")

    def _handle_streaming_response_sync(
        self,
        stream: Generator,
    ) -> Generator[
        tuple[
            StreamingChatResponse | SSEErrorResponse | None,
            AIModelCallResponse | None,
        ]
    ]:
        """Shared streaming logic with async/sync handling"""
        self.streaming_manager = StreamingManager(model_endpoint=self.model_endpoint)

        try:
            for chunk in stream:
                processed_chunks = self.parse_stream_chunk(chunk)
                for pchunk in processed_chunks:
                    yield pchunk, None

            # API has stopped streaming, send done-chunk and get final response
            done_chunk = self.streaming_manager.get_streaming_done_chunk()
            yield done_chunk, None

            final_response = self.streaming_manager.complete()
            logger.debug(f"API has stopped streaming, final_response={final_response}")

            yield None, final_response
            return  # Stop the generator
        except Exception as e:
            logger.exception(f"_handle_streaming_response_sync: Error: {e}")
            error_response = self._create_streaming_error_response(exc=e)
            yield error_response, None

    async def _handle_streaming_response_async(
        self,
        stream: AsyncGenerator,
    ) -> AsyncGenerator[
        tuple[
            StreamingChatResponse | SSEErrorResponse | None,
            AIModelCallResponse | None,
        ]
    ]:
        """Shared streaming logic with async/sync handling"""
        self.streaming_manager = StreamingManager(model_endpoint=self.model_endpoint)

        try:
            async for chunk in stream:
                processed_chunks = self.parse_stream_chunk(chunk)
                for pchunk in processed_chunks:
                    yield pchunk, None

            # API has stopped streaming, send done-chunk and get final response
            done_chunk = self.streaming_manager.get_streaming_done_chunk()
            yield done_chunk, None

            logger.debug("API has stopped streaming, processsing final response")
            final_response = self.streaming_manager.complete()

            yield None, final_response
            return  # Stop the generator
        except Exception as e:
            logger.exception(f"_handle_streaming_response_async: Error: {e}")
            error_response = self._create_streaming_error_response(exc=e)
            yield error_response, None

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
    def get_api_call_params(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        pass

    @abstractmethod
    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        pass

    @abstractmethod
    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        pass

    @abstractmethod
    def do_streaming_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        pass

    @abstractmethod
    async def do_streaming_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        pass

    @abstractmethod
    def parse_response(self, response) -> ChatResponse | ImageResponse | None:
        pass

    @abstractmethod
    def parse_stream_chunk(self, chunk) -> StreamingChatResponse | SSEErrorResponse | None:
        pass

    @abstractmethod
    def _get_usage_from_provider_response(self, response):
        pass

    def validate_inputs(
        self,
        prompt: FormattedPrompt | dict,
        context: list[FormattedPrompt | dict] | None = None,
    ) -> tuple[FormattedPrompt, list[FormattedPrompt] | None] | None:
        try:
            validated_prompt = self._validate_prompt(prompt)
            validated_context = [self._validate_prompt(pmt) for pmt in context]

            if not self.validate_options():
                logger.error("validate_inputs: ERROR: validate_options failed")
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
        elif (
            isinstance(prompt, str)
            and self.model_endpoint.ai_model.functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION
        ):
            validated = prompt
            return prompt
        else:
            raise ValueError(f"Prompt type {type(prompt)} not valid. prompt={prompt} ")

        return validated.model_dump()

    def validate_options(self) -> bool:
        """Validate configuration options"""
        return self.model_endpoint.ai_model.validate_options(self.config.options)

    # -------------------------------------------------------------------------
    # For Usage and cost

    def get_usage_and_charge(
        self,
        response,
        usage=None,
    ) -> tuple[
        ChatResponseUsage | ImageResponseUsage | None,
        UsageCharge | None,
    ]:
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

    def get_unknown_content_type_item(
        self,
        index: int,
        role: str,
        unknown_item: Any,
        streaming: bool,
    ):
        logger.debug(f"process_content_item_delta: Unknown content item type {type(unknown_item)}")

        item_dict = {
            "index": index,
            "role": role,
            "metadata": {
                "data": unknown_item.model_dump() if hasattr(unknown_item, "model_dump") else str(unknown_item),
                "type": type(unknown_item),
            },
        }
        return (
            ChatResponseGenericContentItemDelta(**item_dict)
            if streaming
            else ChatResponseGenericContentItem(**item_dict)
        )
