import logging
from typing import Any

from openai.types import ImagesResponse as OpenAIImagesResponse

from dhenara.ai.providers.openai import OpenAIClientBase
from dhenara.ai.types.external_api import (
    AIModelAPIProviderEnum,
    SystemInstructions,
)
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    ImageContentFormat,
    ImageResponse,
    ImageResponseChoice,
    ImageResponseContentItem,
    ImageResponseUsage,
    StreamingChatResponse,
)
from dhenara.ai.types.shared.api import SSEErrorResponse

logger = logging.getLogger(__name__)


class OpenAIImage(OpenAIClientBase):
    """OpenAI Image Generation Client"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional params
        self.response_format = None

    def get_api_call_params(
        self,
        prompt: str,
        context: list[str] | None = None,
        instructions: SystemInstructions | None = None,
    ) -> AIModelCallResponse:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("inputs must be validated with `self.validate_inputs()` before api calls")

        instructions_str = " ".join(instructions)
        prompt_text = f"{instructions_str} {context} {prompt}"
        model_options = self.model_endpoint.ai_model.get_options_with_defaults(self.config.options)
        user = self.config.get_user()

        image_args: dict[str, Any] = {
            "model": self.model_name_in_api_calls,
            "prompt": prompt_text,
            **model_options,
        }

        if user:
            if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
                image_args["user"] = user

        # Store additional params
        self.response_format = model_options["response_format"]  # Special case

        return {"image_args": image_args}

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        image_args = api_call_params["image_args"]
        if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            response = self._client.images.generate(**image_args)
        else:
            response = self._client.complete(**image_args)  # Images on Azure NOT tested

        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        image_args = api_call_params["image_args"]
        if self.model_endpoint.api.provider != AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            response = await self._client.images.generate(**image_args)
        else:
            response = await self._client.complete(**image_args)  # Images on Azure NOT tested
        return response

    def do_streaming_api_call_sync(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        raise ValueError("do_streaming_api_call_sync:  Streaming not supported for Image generation")

    async def do_streaming_api_call_async(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        raise ValueError("do_streaming_api_call_async:  Streaming not supported for Image generation")

    def parse_stream_chunk(
        self,
        chunk,
    ) -> StreamingChatResponse | SSEErrorResponse | None:
        raise ValueError("parse_stream_chunk: Streaming not supported for Image generation")

    def _get_usage_from_provider_response(
        self,
        response: OpenAIImagesResponse,
    ) -> ImageResponseUsage:
        # No usage data availabe in response. We will derive some params
        model = self.model_endpoint.ai_model.model_name
        model_options = self.model_endpoint.ai_model.get_options_with_defaults(self.config.options)

        return ImageResponseUsage(
            number_of_images=len(response.data),
            model=model,
            options=model_options,
        )

    def parse_response(
        self,
        response: OpenAIImagesResponse,
    ) -> ImageResponse:
        """Parse OpenAI image response into standard format"""

        usage, usage_charge = self.get_usage_and_charge(response)

        choices = []
        for idx, image in enumerate(response.data):
            if self.response_format == "b64_json":
                choices.append(
                    ImageResponseChoice(
                        index=idx,
                        contents=[
                            ImageResponseContentItem(
                                index=0,
                                content_format=ImageContentFormat.BASE64,
                                content_b64_json=image.b64_json,
                                metadata={
                                    "revised_prompt": image.revised_prompt,
                                },
                            )
                        ],
                    )
                )

            elif self.response_format == "url":
                choices.append(
                    ImageResponseChoice(
                        index=idx,
                        contents=[
                            ImageResponseContentItem(
                                index=0,
                                content_format=ImageContentFormat.URL,
                                content_url=image.url,
                                metadata={
                                    "revised_prompt": image.revised_prompt,
                                },
                            )
                        ],
                    )
                )
            else:
                raise ValueError(f"Unknown response_format {self.response_format} in parse_response:")

        return ImageResponse(
            model=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            choices=choices,
            usage=usage,
            usage_charge=usage_charge,
        )
