import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from google.genai.types import GenerateImagesConfig, GenerateImagesResponse

from dhenara.ai.providers.common.message_text import build_image_prompt_text
from dhenara.ai.providers.google import GoogleAIClientBase
from dhenara.ai.types.genai import (
    ImageContentFormat,
    ImageResponse,
    ImageResponseChoice,
    ImageResponseContentItem,
    ImageResponseUsage,
)
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt, SystemInstruction

logger = logging.getLogger(__name__)


class GoogleAIImage(GoogleAIClientBase):
    """GoogleAI Image Generation Client"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_api_call_params(
        self,
        prompt: str | dict | Prompt | None,
        context: Sequence[str | dict | Prompt] | None = None,
        instructions: dict[str, Any] | list[str | dict | SystemInstruction] | None = None,
        messages: Sequence[MessageItem] | None = None,
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("Inputs must be validated before API calls")

        prompt_text = build_image_prompt_text(
            prompt=prompt,
            context=context,
            instructions=instructions,
            messages=messages,
            formatter=self.formatter,
        )

        generate_config_args = self.get_default_generate_config_args()
        generate_config = GenerateImagesConfig(**generate_config_args)

        return {
            "prompt": prompt_text,
            "generate_config": generate_config,
        }

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> GenerateImagesResponse:
        client = self._client
        if client is None:
            raise RuntimeError("Google client not initialized")
        response = client.models.generate_images(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            prompt=api_call_params["prompt"],
        )
        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> GenerateImagesResponse:
        client = self._client
        if client is None:
            raise RuntimeError("Google client not initialized")
        response = await client.models.generate_images(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            prompt=api_call_params["prompt"],
        )
        return response

    def do_streaming_api_call_sync(
        self,
        api_call_params: dict[str, Any],
    ) -> Iterator[object]:
        raise ValueError("do_streaming_api_call_sync:  Streaming not supported for Image generation")

    async def do_streaming_api_call_async(
        self,
        api_call_params: dict[str, Any],
    ) -> AsyncIterator[object]:
        raise ValueError("do_streaming_api_call_async:  Streaming not supported for Image generation")

    def get_default_generate_config_args(self) -> dict:
        model_options = self.model_endpoint.ai_model.get_options_with_defaults(self.config.options)

        config_params = {
            **model_options,
            "output_mime_type": "image/jpeg",
            "include_rai_reason": True,
            "safety_filter_level": "block_only_high",
            # "person_generation": "allow_adult",
            # "negative_prompt":"Outside","human"
        }

        return config_params

    def parse_stream_chunk(
        self,
        chunk,
    ):
        raise ValueError("parse_stream_chunk: Streaming not supported for Image generation")

    def _get_usage_from_provider_response(
        self,
        response: GenerateImagesResponse,
    ) -> ImageResponseUsage:
        # No usage data availabe in response. We will derive some params
        model = self.model_endpoint.ai_model.model_name
        model_options = self.model_endpoint.ai_model.get_options_with_defaults(self.config.options)
        images = response.generated_images or []

        return ImageResponseUsage(
            number_of_images=len(images),
            model=model,
            options=model_options,
        )

    def parse_response(
        self,
        response: GenerateImagesResponse,
    ) -> ImageResponse:
        """Parse GoogleAI image response into standard format"""

        usage, usage_charge = self.get_usage_and_charge(response)
        usage_img = usage if isinstance(usage, ImageResponseUsage) else None
        images = response.generated_images or []
        choices = []
        for idx, image in enumerate(images):
            image_obj = getattr(image, "image", None)
            img_bytes = getattr(image_obj, "image_bytes", None) if image_obj is not None else None
            if img_bytes is None:
                continue
            choices.append(
                ImageResponseChoice(
                    index=idx,
                    contents=[
                        ImageResponseContentItem(
                            index=0,
                            content_format=ImageContentFormat.BYTES,
                            content_bytes=img_bytes,
                            metadata={
                                "rai_filtered_reason": image.rai_filtered_reason,
                            },
                        )
                    ],
                )
            )

        return ImageResponse(
            model=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            choices=choices,
            usage=usage_img,
            usage_charge=usage_charge,
        )
