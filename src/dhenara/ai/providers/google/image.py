import logging

from google.genai.types import GenerateImagesConfig, GenerateImagesResponse

from dhenara.ai.providers.google import GoogleAIClientBase
from dhenara.ai.types.external_api import SystemInstructions
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    ImageContentFormat,
    ImageResponse,
    ImageResponseChoice,
    ImageResponseContentItem,
    ImageResponseUsage,
)

logger = logging.getLogger(__name__)


class GoogleAIImage(GoogleAIClientBase):
    """GoogleAI Image Generation Client"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_api_call_params(
        self,
        prompt: str,
        context: list[str] | None = None,
        instructions: SystemInstructions | None = None,
    ) -> AIModelCallResponse:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("Inputs must be validated before API calls")

        instructions_str = " ".join(instructions)
        prompt_text = f"{instructions_str} {context} {prompt}"

        generate_config_args = self.get_default_generate_config_args()
        generate_config = GenerateImagesConfig(**generate_config_args)

        ## Process instructions
        # instructions_str = self.process_instructions(instructions)
        # if isinstance(instructions_str, dict):
        #    if context:
        #        context.insert(0, instructions_str)
        #    else:
        #        context = [instructions_str]
        # elif instructions_str and not any(model_name.startswith(model) for model in ["gemini-1.0-pro"]):
        #    generate_config.system_instruction = instructions_str

        return {
            "prompt": prompt_text,
            "history": context or [],
            "generate_config": generate_config,
        }

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        response = self._client.models.generate_images(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            prompt=api_call_params["prompt"],
        )
        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        response = await self._client.models.generate_images(
            model=self.model_name_in_api_calls,
            config=api_call_params["generate_config"],
            prompt=api_call_params["prompt"],
        )
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

        return ImageResponseUsage(
            number_of_images=len(response.generated_images),
            model=model,
            options=model_options,
        )

    def parse_response(
        self,
        response: GenerateImagesResponse,
    ) -> ImageResponse:
        """Parse GoogleAI image response into standard format"""

        usage, usage_charge = self.get_usage_and_charge(response)
        choices = []
        for idx, image in enumerate(response.generated_images):
            choices.append(
                ImageResponseChoice(
                    index=idx,
                    contents=[
                        ImageResponseContentItem(
                            index=0,
                            content_format=ImageContentFormat.BYTES,
                            content_bytes=image.image.image_bytes,
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
            usage=usage,
            usage_charge=usage_charge,
        )
