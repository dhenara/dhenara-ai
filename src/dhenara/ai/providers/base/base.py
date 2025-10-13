import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator, Sequence
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
    ExternalApiCallStatus,
    ExternalApiCallStatusEnum,
    ImageResponse,
    ImageResponseUsage,
    StreamingChatResponse,
    UsageCharge,
)
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt, SystemInstruction
from dhenara.ai.types.shared.api import SSEErrorCode, SSEErrorData, SSEErrorResponse
from dhenara.ai.utils.artifacts import ArtifactWriter

logger = logging.getLogger(__name__)


class AIModelProviderClientBase(ABC):
    """Base class for AI model provider handlers"""

    formatter = None

    def __init__(self, model_endpoint: AIModelEndpoint, config: AIModelCallConfig, is_async: bool = True):
        self.model_endpoint = model_endpoint
        self.model_name_in_api_calls = self.model_endpoint.ai_model.model_name_with_version_suffix
        self.config = config
        self.is_async = is_async
        self._client = None
        self._initialized = False
        self._input_validation_pending = True
        self.streaming_manager = None
        self.formatted_config = None

        if self.formatter is None:
            raise ValueError("Formatter is not set")

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

    def _get_client_http_params(self, api=None) -> tuple[str, dict]:
        params = {}
        if self.config.timeout:
            params["timeout"] = self.config.timeout

        return params

    def _serialize_dhenara_request(
        self,
        prompt: str | dict | Prompt | None,
        context: Sequence[str | dict | Prompt] | None,
        instructions: list[str | dict | SystemInstruction] | None,
        messages: Sequence[MessageItem] | None,
    ) -> dict:
        """Serialize dhenara request format for artifact capture."""
        return {
            "prompt": prompt.model_dump() if isinstance(prompt, Prompt) else prompt,
            "context": [c.model_dump() if isinstance(c, Prompt) else c for c in (context or [])],
            "instructions": [i.model_dump() if isinstance(i, SystemInstruction) else i for i in (instructions or [])],
            "messages": [m.model_dump() if isinstance(m, MessageItem) else m for m in (messages or [])],
            "config": self.config.model_dump() if self.config else {},
        }

    def _capture_artifacts(
        self,
        stage: str,
        data: Any,
        filename: str,
    ) -> None:
        """Capture artifacts if enabled in config.

        Artifacts are written to <artifact_root>/<prefix>/dai/ subdirectory.
        """
        if not self.config.artifact_config or not self.config.artifact_config.enabled:
            return

        artifact_config = self.config.artifact_config

        # Check if this stage should be captured
        if stage == "dhenara_request" and not artifact_config.capture_dhenara_request:
            return
        if stage == "provider_request" and not artifact_config.capture_provider_request:
            return
        if stage == "provider_response" and not artifact_config.capture_provider_response:
            return
        if stage == "dhenara_response" and not artifact_config.capture_dhenara_response:
            return

        # Build prefix: combine user prefix with 'dai' subdirectory
        combined_prefix = f"{artifact_config.prefix}/dai" if artifact_config.prefix else "dai"

        ArtifactWriter.write_json(
            artifact_root=artifact_config.artifact_root,
            filename=filename,
            data=data,
            prefix=combined_prefix,
        )

    def generate_response_sync(
        self,
        prompt: str | dict | Prompt | None,
        context: Sequence[str | dict | Prompt] | None = None,
        instructions: list[str | dict | SystemInstruction] | None = None,
        messages: Sequence[MessageItem] | None = None,
    ) -> AIModelCallResponse:
        parsed_response: ChatResponse | None = None
        api_call_status: ExternalApiCallStatus | None = None

        logger.debug(f"generate_response: prompt={prompt}, context={context}")

        # Capture dhenara request format
        self._capture_artifacts(
            stage="dhenara_request",
            data=self._serialize_dhenara_request(prompt, context, instructions, messages),
            filename="dai_request.json",
        )

        api_call_params = self.get_api_call_params(
            prompt=prompt,
            context=context,
            instructions=instructions,
            messages=messages,
        )

        logger.debug(f"generate_response: api_call_params: {api_call_params}")

        # Capture provider request format (already in provider-specific format)
        self._capture_artifacts(
            stage="provider_request",
            data=api_call_params,
            filename="dai_provider_request.json",
        )

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
                # Return the provider streaming generator; artifacts will be captured
                # inside the streaming handler when the final aggregated response is produced.
                return AIModelCallResponse(sync_stream_generator=stream_generator)

            response = self.do_api_call_sync(api_call_params)

            # Capture provider response
            self._capture_artifacts(
                stage="provider_response",
                data=response if isinstance(response, dict) else str(response),
                filename="dai_provider_response.json",
            )

            parsed_response = self.parse_response(response)
            api_call_status = self._create_success_status()
            ai_response = self._get_ai_model_call_response(parsed_response, api_call_status)

            # Capture dhenara response format
            self._capture_artifacts(
                stage="dhenara_response",
                data=ai_response.model_dump() if hasattr(ai_response, "model_dump") else str(ai_response),
                filename="dai_response.json",
            )

            return ai_response

        except Exception as e:
            logger.exception(f"Error in generate_response_sync: {e}")
            api_call_status = self._create_error_status(str(e))

    async def generate_response_async(
        self,
        prompt: str | dict | Prompt | None,
        context: Sequence[str | dict | Prompt] | None = None,
        instructions: list[str | dict | SystemInstruction] | None = None,
        messages: Sequence[MessageItem] | None = None,
    ) -> AIModelCallResponse:
        parsed_response: ChatResponse | None = None
        api_call_status: ExternalApiCallStatus | None = None

        logger.debug(f"generate_response: prompt={prompt}, context={context}")

        # Capture dhenara request format
        self._capture_artifacts(
            stage="dhenara_request",
            data=self._serialize_dhenara_request(prompt, context, instructions, messages),
            filename="dai_request.json",
        )

        api_call_params = self.get_api_call_params(
            prompt=prompt,
            context=context,
            instructions=instructions,
            messages=messages,
        )

        logger.debug(f"generate_response: api_call_params: {api_call_params}")

        # Capture provider request format (already in provider-specific format)
        self._capture_artifacts(
            stage="provider_request",
            data=api_call_params,
            filename="dai_provider_request.json",
        )

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
                # Return the provider streaming generator; artifacts will be captured
                # inside the streaming handler when the final aggregated response is produced.
                return AIModelCallResponse(async_stream_generator=stream_generator)

            response = await self.do_api_call_async(api_call_params)

            # Capture provider response
            self._capture_artifacts(
                stage="provider_response",
                data=response if isinstance(response, dict) else str(response),
                filename="dai_provider_response.json",
            )

            parsed_response = self.parse_response(response)
            api_call_status = self._create_success_status()
            ai_response = self._get_ai_model_call_response(parsed_response, api_call_status)

            # Capture dhenara response format
            self._capture_artifacts(
                stage="dhenara_response",
                data=ai_response.model_dump() if hasattr(ai_response, "model_dump") else str(ai_response),
                filename="dai_response.json",
            )

            return ai_response

        except Exception as e:
            logger.exception(f"Error in generate_response_async: {e}")
            api_call_status = self._create_error_status(str(e))

    def _format_and_generate_response_sync(
        self,
        prompt: str | dict | Prompt | None,
        context: Sequence[str | dict | Prompt] | None = None,
        instructions: list[str | dict | SystemInstruction] | None = None,
        messages: Sequence[MessageItem] | None = None,
    ) -> AIModelCallResponse:
        """Generate response from the model"""

        if settings.ENABLE_INPUT_FORMAT_CONVERSION:
            formatted_inputs = self.format_inputs(
                prompt=prompt,
                context=context,
                instructions=instructions,
                messages=messages,
            )
            if not formatted_inputs:
                return AIModelCallResponse(
                    status=self._create_error_status(
                        message="Input validation failed, not proceeding to generation.",
                        status=ExternalApiCallStatusEnum.REQUEST_NOT_SEND,
                    )
                )
            return self.generate_response_sync(
                prompt=formatted_inputs["prompt"],
                context=formatted_inputs["context"],
                instructions=formatted_inputs["instructions"],
                messages=formatted_inputs.get("messages"),
            )
        else:
            self._input_validation_pending = False
            return self.generate_response_sync(
                prompt=prompt,
                context=context,
                instructions=instructions,
                messages=messages,
            )

    async def _format_and_generate_response_async(
        self,
        prompt: str | dict | Prompt | None,
        context: Sequence[str | dict | Prompt] | None = None,
        instructions: list[str | dict | SystemInstruction] | None = None,
        messages: Sequence[MessageItem] | None = None,
    ) -> AIModelCallResponse:
        """Generate response from the model"""

        if settings.ENABLE_INPUT_FORMAT_CONVERSION:
            formatted_inputs = self.format_inputs(
                prompt=prompt,
                context=context,
                instructions=instructions,
                messages=messages,
            )
            if not formatted_inputs:
                return AIModelCallResponse(
                    status=self._create_error_status(
                        message="Input validation failed, not proceeding to generation.",
                        status=ExternalApiCallStatusEnum.REQUEST_NOT_SEND,
                    )
                )
            return await self.generate_response_async(
                prompt=formatted_inputs["prompt"],
                context=formatted_inputs["context"],
                instructions=formatted_inputs["instructions"],
                messages=formatted_inputs.get("messages"),
            )
        else:
            self._input_validation_pending = False
            return await self.generate_response_async(
                prompt=prompt,
                context=context,
                instructions=instructions,
                messages=messages,
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
            logger.debug("API has stopped streaming; capturing artifacts and yielding final response")

            # Capture provider-level streaming summary as provider_response
            try:
                # Persist a compact provider_response built from consolidated choices and metadata
                chat = final_response.chat_response
                provider_payload = {
                    "usage": (chat.usage.model_dump() if chat and chat.usage else None),
                    "usage_charge": (chat.usage_charge.model_dump() if chat and chat.usage_charge else None),
                    "choices": ([c.model_dump() for c in (chat.choices or [])] if chat else []),
                    "metadata": (chat.metadata.model_dump() if chat and chat.metadata else {}),
                }
                self._capture_artifacts(
                    stage="provider_response",
                    data=provider_payload,
                    filename="dai_provider_response.json",
                )
            except Exception as e:
                logger.debug(f"Streaming artifact capture (provider_response) failed: {e}")

            # Capture the final Dhenara response wrapper
            try:
                self._capture_artifacts(
                    stage="dhenara_response",
                    data=final_response.model_dump() if hasattr(final_response, "model_dump") else str(final_response),
                    filename="dai_response.json",
                )
            except Exception as e:
                logger.debug(f"Streaming artifact capture (dhenara_response) failed: {e}")

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

            logger.debug("API has stopped streaming; capturing artifacts and yielding final response")
            final_response = self.streaming_manager.complete()

            # Capture provider-level streaming summary as provider_response
            try:
                chat = final_response.chat_response
                provider_payload = {
                    "usage": (chat.usage.model_dump() if chat and chat.usage else None),
                    "usage_charge": (chat.usage_charge.model_dump() if chat and chat.usage_charge else None),
                    "choices": ([c.model_dump() for c in (chat.choices or [])] if chat else []),
                    "metadata": (chat.metadata.model_dump() if chat and chat.metadata else {}),
                }
                self._capture_artifacts(
                    stage="provider_response",
                    data=provider_payload,
                    filename="dai_provider_response.json",
                )
            except Exception as e:
                logger.debug(f"Streaming artifact capture (provider_response) failed: {e}")

            # Capture the final Dhenara response wrapper
            try:
                self._capture_artifacts(
                    stage="dhenara_response",
                    data=final_response.model_dump() if hasattr(final_response, "model_dump") else str(final_response),
                    filename="dai_response.json",
                )
            except Exception as e:
                logger.debug(f"Streaming artifact capture (dhenara_response) failed: {e}")

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
    def get_api_call_params(
        self,
        prompt: dict | None,
        context: list[dict] | None = None,
        instructions: dict | None = None,
        messages: Sequence[MessageItem] | None = None,
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

    def format_inputs(
        self,
        prompt: str | dict | Prompt | None,
        context: Sequence[str | dict | Prompt] | None = None,
        instructions: list[str | dict | SystemInstruction] | None = None,
        messages: Sequence[MessageItem] | None = None,
        **kwargs,
    ) -> tuple[dict, list[dict], list[str] | dict | None]:
        """Format inputs into provider-specific formats"""
        try:
            # Validate mutual exclusivity: either (prompt+context) OR messages, not both
            has_traditional_inputs = prompt is not None or (context is not None and len(context) > 0)
            has_messages = messages is not None and len(messages) > 0

            if has_traditional_inputs and has_messages:
                raise ValueError(
                    "Cannot use both 'messages' and 'prompt/context' parameters. "
                    "Use either messages for multi-turn conversations or prompt/context for traditional inputs."
                )

            if not has_traditional_inputs and not has_messages:
                raise ValueError("Either 'messages' or 'prompt/context' must be provided")

            # If messages are provided, format them and instructions
            # Provider-specific formatting will happen in get_api_call_params
            if has_messages:
                # Format instructions if provided
                formatted_instructions = None
                if instructions:
                    formatted_instructions = self.formatter.format_instructions(
                        instructions=instructions,
                        model_endpoint=self.model_endpoint,
                        **kwargs,
                    )

                # Validate Options
                if not self.model_endpoint.ai_model.validate_options(self.config.options):
                    raise ValueError("validate_inputs: ERROR: validate_options failed")

                self.formatted_config = self.config
                self._input_validation_pending = False

                return {
                    "prompt": None,
                    "context": [],
                    "instructions": formatted_instructions,
                    "messages": messages,
                }

            # Traditional path: format prompt and context
            # Format prompt
            formatted_prompt = self.formatter.format_prompt(
                prompt=prompt,
                model_endpoint=self.model_endpoint,
                **kwargs,
            )

            # Format context
            formatted_context = []
            if context:
                formatted_context = self.formatter.format_context(
                    context=context,
                    model_endpoint=self.model_endpoint,
                    **kwargs,
                )

            # Format instructions
            formatted_instructions = None
            if instructions:
                formatted_instructions = self.formatter.format_instructions(
                    instructions=instructions,
                    model_endpoint=self.model_endpoint,
                    **kwargs,
                )

            # Validate Options
            if not self.model_endpoint.ai_model.validate_options(self.config.options):
                raise ValueError("validate_inputs: ERROR: validate_options failed")

            # Options validation successful.
            # TODO_FUTURE
            # Clone the config, and convert t relevant fields to provider format
            # self.formatted_config = self.config.model_copy()
            self.formatted_config = self.config

            self._input_validation_pending = False
            return {
                "prompt": formatted_prompt,
                "context": formatted_context,
                "instructions": formatted_instructions,
            }

        except Exception as e:
            logger.exception(f"format_inputs: {e}")
            return None, None, None

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
        logger.debug(f"Unknown content item type {type(unknown_item)}")

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
