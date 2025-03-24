import logging

from google import genai

from dhenara.ai.providers.base import AIModelProviderClientBase
from dhenara.ai.providers.shared import APIProviderSharedFns
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum

from .formatter import GoogleFormatter

logger = logging.getLogger(__name__)


class GoogleAIClientBase(AIModelProviderClientBase):
    """Base class for all Google AI Clients"""

    formatter = GoogleFormatter

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def _get_client_params(self, api) -> tuple[str, dict]:
        """Common logic for both sync and async clients"""
        if api.provider == AIModelAPIProviderEnum.GOOGLE_AI:
            return "google_ai", {
                "api_key": api.api_key,
                # "http_options": {"api_version": "v1beta"},
            }
        elif api.provider == AIModelAPIProviderEnum.GOOGLE_VERTEX_AI:
            client_params = APIProviderSharedFns.get_vertex_ai_credentials(api)
            return "vertex_ai", {
                "vertexai": True,
                "credentials": client_params["credentials"],
                "project": client_params["project_id"],
                "location": client_params["location"],
                # "http_options": {"api_version": "v1beta"},
            }
        else:
            error_msg = f"Unsupported API provider {api.provider} for Google AI"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _setup_client_sync(self) -> genai.Client:
        """Get the appropriate sync Google AI client"""
        api = self.model_endpoint.api
        client_type, params = self._get_client_params(api)
        return genai.Client(**params)

    async def _setup_client_async(self) -> genai.Client:
        """Get the appropriate async Google AI client"""
        api = self.model_endpoint.api
        client_type, params = self._get_client_params(api)
        return genai.Client(**params).aio
