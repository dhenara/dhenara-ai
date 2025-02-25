import logging
from typing import Any

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicVertex
from dhenara.ai.providers.base import AIModelProviderClientBase
from dhenara.ai.providers.shared import APIProviderSharedFns
from dhenara.ai.types.external_api import (
    AIModelAPIProviderEnum,
    AnthropicMessageRoleEnum,
    AnthropicPromptMessage,
    FormattedPrompt,
    SystemInstructions,
)
from dhenara.ai.types.genai import AIModel
from dhenara.ai.types.shared.file import FileFormatEnum, GenericFile

logger = logging.getLogger(__name__)


class AnthropicClientBase(AIModelProviderClientBase):
    """Base class for all Anthropic Clients"""

    prompt_message_class = AnthropicPromptMessage

    async def initialize(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    def process_instructions(
        self,
        instructions: SystemInstructions,
    ) -> FormattedPrompt | str | None:
        if instructions:
            if isinstance(instructions, list):
                return " ".join(instructions)
            logger.warning(f"process_instructions: instructions should be a list not {type(instructions)}")
            return str(instructions)
        return None

    async def _setup_client(self) -> AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicVertex:
        """Get the appropriate Anthropic client based on the provider"""
        api = self.model_endpoint.api

        if api.provider == AIModelAPIProviderEnum.ANTHROPIC:
            return AsyncAnthropic(api_key=api.api_key)

        elif api.provider == AIModelAPIProviderEnum.GOOGLE_VERTEX_AI:
            client_params = APIProviderSharedFns.get_vertex_ai_credentials(api)

            return AsyncAnthropicVertex(
                credentials=client_params["credentials"],
                project_id=client_params["project_id"],
                region=client_params["location"],
            )

        elif api.provider == AIModelAPIProviderEnum.AMAZON_BEDROCK:
            client_params = api.get_provider_credentials()
            return AsyncAnthropicBedrock(
                aws_access_key=client_params["aws_access_key"],
                aws_secret_key=client_params["aws_secret_key"],
                aws_region=client_params.get("aws_region", "us-east-1"),
            )

        else:
            error_msg = f"Unsupported API provider {api.provider} for Anthropic"
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def get_prompt(
        model: AIModel,
        role: AnthropicMessageRoleEnum,
        text: str,
        file_contents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if file_contents:
            # Combine text and file contents into a single content array
            content = [
                {
                    "type": "text",
                    "text": text,
                },
                *file_contents,
            ]
        else:
            # For text-only messages, content should be a string
            content = text

        return {"role": role.value, "content": content}

    @staticmethod
    def get_prompt_file_contents(
        model: AIModel,
        files: list[GenericFile],
        max_words: int | None,
    ) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for file in files:
            file_format = file.get_file_format()
            try:
                if file_format in [FileFormatEnum.COMPRESSED, FileFormatEnum.TEXT]:
                    text = f"\nFile: {file.get_source_file_name()}  Content: {file.get_processed_file_data(max_words=max_words)}"
                    contents.append(
                        {
                            "type": "text",
                            "text": text,
                        }
                    )
                elif file_format == FileFormatEnum.IMAGE:
                    mime_type = file.get_mime_type()
                    if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                        raise ValueError(f"Unsupported media type: {mime_type}")

                    contents.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": file.get_processed_file_data_content_only(),
                            },
                        }
                    )
                else:
                    logger.error(f"Unknown file_format {file_format} for file {file.name}")
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {e}")

        return contents
