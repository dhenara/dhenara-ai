import logging
from typing import Any

from dhenara.ai.providers.base import AIModelProviderClientBase
from dhenara.ai.types.external_api import (
    AIModelAPIProviderEnum,
    AIModelFunctionalTypeEnum,
    FormattedPrompt,
    OpenAiMessageRoleEnum,
    OpenAIPromptMessage,
    SystemInstructions,
)
from dhenara.ai.types.genai import AIModel
from dhenara.ai.types.shared.file import FileFormatEnum, GenericFile
from openai import AsyncAzureOpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
class OpenAIClientBase(AIModelProviderClientBase):
    """Base class for all OpenAI Clients"""

    prompt_message_class = OpenAIPromptMessage

    async def initialize(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    def process_instructions(
        self,
        instructions: SystemInstructions,
    ) -> FormattedPrompt | str | None:
        instructions_str = None
        if instructions:
            if isinstance(instructions, list):
                instructions_str = " ".join(instructions)
            else:
                logger.warning(f"get_model_response: instructions should be a list not {type(instructions)}")
                instructions_str = str(instructions)

            # Beta models won't support system role
            system_role = OpenAiMessageRoleEnum.USER if self.model_endpoint.ai_model.beta else OpenAiMessageRoleEnum.SYSTEM
            instruction_as_prompt = self.get_prompt(
                model=self.model_endpoint.ai_model,
                role=system_role,
                text=instructions_str,
                file_contents=[],
            )
            return instruction_as_prompt
        return instructions_str

    async def _setup_client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        """Get the appropriate async OpenAI client based on the provider"""
        api = self.model_endpoint.api

        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.OPEN_AI:
            return AsyncOpenAI(api_key=api.api_key)
        elif api.provider == AIModelAPIProviderEnum.MICROSOFT_OPENAI:
            client_params = api.get_provider_credentials()

            return AsyncAzureOpenAI(
                api_key=client_params["api_key"],
                azure_endpoint=client_params["azure_endpoint"],
                api_version=client_params["api_version"],
            )

        elif api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            from azure.ai.inference.aio import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            client_params = api.get_provider_credentials()

            return ChatCompletionsClient(
                endpoint=client_params["azure_endpoint"],
                credential=AzureKeyCredential(
                    key=client_params["api_key"],
                ),
            )
        else:
            error_msg = f"Unsupported API provider {api.provider} for OpenAI functions"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # -------------------------------------------------------------------------
    # Static methods
    # -------------------------------------------------------------------------
    @staticmethod
    def get_prompt(
        model: AIModel,
        role: OpenAiMessageRoleEnum,
        text: str,
        file_contents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if model.functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
            return OpenAIClientBase.get_prompt_image_model(
                model=model,
                role=role,
                text=text,
                file_contents=file_contents,
            )

        if file_contents:
            content = [
                {
                    "type": "text",
                    "text": text,
                },
                *file_contents,
            ]
        else:
            content = text

        return {"role": role.value, "content": content}

    @staticmethod
    def get_prompt_image_model(
        model: AIModel | None,
        role: OpenAiMessageRoleEnum | None,
        text: str,
        file_contents: list[dict[str, Any]],
    ) -> str:
        if file_contents:
            _file_content = " ".join(file_contents)
            content = text + " " + _file_content
        else:
            content = text

        return content

    # -------------------------------------------------------------------------
    @staticmethod
    def get_prompt_file_contents(
        model: AIModel,
        files: list[GenericFile],
        max_words: int | None,
    ) -> list[dict[str, Any]]:
        if model.functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
            return OpenAIClientBase.get_prompt_file_contents_image_model(
                model=model,
                files=files,
                max_words=max_words,
            )

        # Eg:
        #        {"type": "text", "text": "What's in this image?"},
        #        {
        #            "type": "image_url",
        #            "image_url": {
        #                "url": "https://upload.wikimedia.org/..boardwalk.jpg",
        #                "url":  f"data:image/jpeg;base64,{base64_image}"
        #            }
        #        },
        contents = []
        for file in files:
            file_format = file.get_file_format()
            try:
                if file_format in [FileFormatEnum.COMPRESSED, FileFormatEnum.TEXT]:
                    text = f"\nFile: {file.get_source_file_name()}  Content: {file.get_processed_file_data(max_words)}"
                    pcontent = {
                        "type": "text",
                        "text": text,
                    }
                    contents.append(pcontent)
                elif file_format in [FileFormatEnum.IMAGE]:
                    mime_type = file.get_mime_type()
                    if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                        raise ValueError(f"Unsupported media type: {mime_type} for file {file.name}")

                    pcontent = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{file.get_processed_file_data_content_only()}",
                            # "url": f"data:image/jpeg;base64,{file.signed_url}",
                        },
                    }
                    contents.append(pcontent)
                else:
                    logger.error(f"get_prompt_file_contents: Unknown file_format {file_format} for file {file.name} ")
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {e}")

        return contents

    # -------------------------------------------------------------------------
    @staticmethod
    def get_prompt_file_contents_image_model(
        model: AIModel,
        files: list[GenericFile],
        max_words: int | None,
    ) -> str:
        contents: list[dict[str, Any]] = []
        for file in files:
            file_format = file.get_file_format()
            try:
                if file_format in [FileFormatEnum.COMPRESSED, FileFormatEnum.TEXT]:
                    text = f"\nFile: {file.get_source_file_name()}  Content: {file.get_processed_file_data(max_words=max_words)}"
                    contents.append(text)
                elif file_format == FileFormatEnum.IMAGE:
                    mime_type = file.get_mime_type()
                    if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                        raise ValueError(f"Unsupported media type: {mime_type} for file {file.name}")

                    pcontent = f"data:{mime_type};base64,{file.get_processed_file_data_content_only()}"
                    contents.append(pcontent)
                else:
                    logger.error(f"get_prompt_file_contents: Unknown file_format {file_format} for file {file.name}")
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {e}")

        return " ".join(contents)
