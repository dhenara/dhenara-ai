import logging
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

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

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
class OpenAIClientBase(AIModelProviderClientBase):
    """Base class for all OpenAI Clients"""

    prompt_message_class = OpenAIPromptMessage

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
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
            system_role = (
                OpenAiMessageRoleEnum.USER if self.model_endpoint.ai_model.beta else OpenAiMessageRoleEnum.SYSTEM
            )
            instruction_as_prompt = self.get_prompt(
                model=self.model_endpoint.ai_model,
                role=system_role,
                text=instructions_str,
                file_contents=[],
            )
            return instruction_as_prompt
        return instructions_str

    def _get_client_params(self, api) -> tuple[str, dict]:
        """Common logic for both sync and async clients"""
        if api.provider == AIModelAPIProviderEnum.OPEN_AI:
            return "openai", {"api_key": api.api_key}

        elif api.provider == AIModelAPIProviderEnum.MICROSOFT_OPENAI:
            client_params = api.get_provider_credentials()
            return "azure_openai", {
                "api_key": client_params["api_key"],
                "azure_endpoint": client_params["azure_endpoint"],
                "api_version": client_params["api_version"],
            }

        elif api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            client_params = api.get_provider_credentials()
            return "azure_ai", {
                "endpoint": client_params["azure_endpoint"],
                "credential": client_params["api_key"],
            }

        error_msg = f"Unsupported API provider {api.provider} for OpenAI functions"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _setup_client_sync(self) -> OpenAI | AzureOpenAI:
        """Get the appropriate sync OpenAI client"""
        api = self.model_endpoint.api
        client_type, params = self._get_client_params(api)

        if client_type == "openai":
            return OpenAI(**params)
        elif client_type == "azure_openai":
            return AzureOpenAI(**params)
        else:  # azure_ai
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            return ChatCompletionsClient(
                endpoint=params["endpoint"],
                credential=AzureKeyCredential(key=params["credential"]),
            )

    async def _setup_client_async(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        """Get the appropriate async OpenAI client"""
        api = self.model_endpoint.api
        client_type, params = self._get_client_params(api)

        if client_type == "openai":
            return AsyncOpenAI(**params)
        elif client_type == "azure_openai":
            return AsyncAzureOpenAI(**params)
        else:  # azure_ai
            from azure.ai.inference.aio import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            return ChatCompletionsClient(
                endpoint=params["endpoint"],
                credential=AzureKeyCredential(key=params["credential"]),
            )

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
                    text = (
                        f"\nFile: {file.get_source_file_name()}  "
                        f"Content: {file.get_processed_file_data(max_words=max_words)}"
                    )
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
