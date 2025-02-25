import logging

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
from google.genai.types import GenerateImagesConfig, GenerateImagesResponse

logger = logging.getLogger(__name__)


class GoogleAIImage(GoogleAIClientBase):
    """GoogleAI Image Generation Client"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def generate_response(
        self,
        prompt: str,
        context: list[str] | None = None,
        instructions: SystemInstructions | None = None,
    ) -> AIModelCallResponse:
        """Generate image response from the model"""
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

        if self.config.test_mode:
            from dhenara.ai.providers.common.dummy import DummyAIModelResponseFns

            return await DummyAIModelResponseFns.get_dummy_ai_model_response(
                ai_model_ep=self.model_endpoint,
                streaming=self.config.streaming,
            )

        try:
            response = await self._client.models.generate_images(
                model=self.model_name_in_api_calls,
                config=generate_config,
                prompt=prompt_text,
            )
            # response = await self._client.models.generate_images(
            #    model="imagen-3.0-generate-002",
            #    prompt="Fuzzy bunnies in my kitchen",
            #    config=GenerateImagesConfig(
            #        number_of_images=1,
            #    ),
            # )

            parsed_response = self.parse_response(response)

            return AIModelCallResponse(
                status=self._create_success_status(),
                image_response=parsed_response,
            )

        except Exception as e:
            logger.exception(f"Error in generate_response: {e}")
            return AIModelCallResponse(status=self._create_error_status(str(e)))

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
                    content=ImageResponseContentItem(
                        content_format=ImageContentFormat.BYTES,
                        content_bytes=image.image.image_bytes,
                        metadata={
                            "rai_filtered_reason": image.rai_filtered_reason,
                        },
                    ),
                )
            )

        return ImageResponse(
            model=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            choices=choices,
            usage=usage,
            usage_charge=usage_charge,
        )
