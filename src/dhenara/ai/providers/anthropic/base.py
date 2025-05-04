import logging

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicVertex,
)

from dhenara.ai.providers.base import AIModelProviderClientBase
from dhenara.ai.providers.shared import APIProviderSharedFns
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum

from .formatter import AnthropicFormatter

logger = logging.getLogger(__name__)


class AnthropicClientBase(AIModelProviderClientBase):
    """Base class for all Anthropic Clients"""

    formatter = AnthropicFormatter

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def _get_client_params(self, api) -> tuple[str, dict]:
        """Common logic for both sync and async clients"""
        if api.provider == AIModelAPIProviderEnum.ANTHROPIC:
            params = {
                "api_key": api.api_key,
                **self._get_client_http_params(api),
            }
            return "anthropic", params

        elif api.provider == AIModelAPIProviderEnum.GOOGLE_VERTEX_AI:
            client_params = APIProviderSharedFns.get_vertex_ai_credentials(api)
            params = {
                "credentials": client_params["credentials"],
                "project_id": client_params["project_id"],
                "region": client_params["location"],
                **self._get_client_http_params(api),
            }
            return "vertex_ai", params

        elif api.provider == AIModelAPIProviderEnum.AMAZON_BEDROCK:
            client_params = api.get_provider_credentials()
            params = {
                "aws_access_key": client_params["aws_access_key"],
                "aws_secret_key": client_params["aws_secret_key"],
                "aws_region": client_params.get("aws_region", "us-east-1"),
                **self._get_client_http_params(api),
            }
            return "bedrock", params

        error_msg = f"Unsupported API provider {api.provider} for Anthropic"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _setup_client_sync(self) -> Anthropic | AnthropicBedrock | AnthropicVertex:
        """Get the appropriate sync Anthropic client based on the provider"""
        client_type, params = self._get_client_params(self.model_endpoint.api)

        if client_type == "anthropic":
            return Anthropic(**params)
        elif client_type == "vertex_ai":
            return AnthropicVertex(**params)
        else:  # bedrock
            return AnthropicBedrock(**params)

    async def _setup_client_async(self) -> AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicVertex:
        """Get the appropriate async Anthropic client based on the provider"""
        client_type, params = self._get_client_params(self.model_endpoint.api)

        if client_type == "anthropic":
            return AsyncAnthropic(**params)
        elif client_type == "vertex_ai":
            return AsyncAnthropicVertex(**params)
        else:  # bedrock
            return AsyncAnthropicBedrock(**params)
