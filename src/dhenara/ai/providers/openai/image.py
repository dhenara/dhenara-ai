import logging
from typing import Any

from dhenara.ai.providers.openai import OpenAIClientBase
from dhenara.ai.types.external_api import (
    SystemInstructions,
)
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    ImageContentFormat,
    ImageResponse,
    ImageResponseChoice,
    ImageResponseContentItem,
    ImageResponseUsage,
)
from openai.types import ImagesResponse as OpenAIImagesResponse

logger = logging.getLogger(__name__)


class OpenAIImage(OpenAIClientBase):
    """OpenAI Image Generation Client"""

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
        model_options = self.model_endpoint.ai_model.get_options_with_defaults(self.config.options)

        image_args: dict[str, Any] = {
            "model": self.model_name_in_api_calls,
            "prompt": prompt_text,
            "user": self.config.get_user(),
            **model_options,
        }

        try:
            response = await self._client.images.generate(**image_args)

            parsed_response = self.parse_response(
                response=response,
                response_format=model_options["response_format"],  # Special case
            )

            return AIModelCallResponse(
                status=self._create_success_status(),
                image_response=parsed_response,
            )

        except Exception as e:
            logger.exception(f"Error in generate_response: {e}")
            return AIModelCallResponse(status=self._create_error_status(str(e)))

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
        response_format,  # response_format send in the request is needed for parsing
    ) -> ImageResponse:
        """Parse OpenAI image response into standard format"""

        usage, usage_charge = self.get_usage_and_charge(response)
        choices = []
        for idx, image in enumerate(response.data):
            if response_format == "b64_json":
                choices.append(
                    ImageResponseChoice(
                        index=idx,
                        content=ImageResponseContentItem(
                            content_format=ImageContentFormat.BASE64,
                            content_b64_json=image.b64_json,
                            metadata={
                                "revised_prompt": image.revised_prompt,
                            },
                        ),
                    )
                )

            elif response_format == "url":
                choices.append(
                    ImageResponseChoice(
                        index=idx,
                        content=ImageResponseContentItem(
                            content_format=ImageContentFormat.URL,
                            content_url=image.url,
                            metadata={
                                "revised_prompt": image.revised_prompt,
                            },
                        ),
                    )
                )
            else:
                raise ValueError(f"Unknown response_format {response_format} in parse_response:")

        return ImageResponse(
            model=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            choices=choices,
            usage=usage,
            usage_charge=usage_charge,
        )
