import logging

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from dhenara.ai.providers.base import AIModelProviderClientBase
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum

from .formatter import OpenAIFormatter

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
class OpenAIClientBase(AIModelProviderClientBase):
    """Base class for all OpenAI Clients"""

    formatter = OpenAIFormatter

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def _get_client_params(self, api) -> tuple[str, dict]:
        """Common logic for both sync and async clients"""
        if api.provider == AIModelAPIProviderEnum.OPEN_AI:
            params = {
                "api_key": api.api_key,
                **self._get_client_http_params(api),
            }
            return "openai", params

        elif api.provider == AIModelAPIProviderEnum.MICROSOFT_OPENAI:
            client_params = api.get_provider_credentials()
            params = {
                "api_key": client_params["api_key"],
                "azure_endpoint": client_params["azure_endpoint"],
                "api_version": client_params["api_version"],
                **self._get_client_http_params(api),
            }
            return "azure_openai", params

        elif api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            client_params = api.get_provider_credentials()
            params = {
                "endpoint": client_params["azure_endpoint"],
                "credential": client_params["api_key"],
                **self._get_client_http_params(api),
            }
            return "azure_ai", params

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
