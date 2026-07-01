import logging
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI, OpenAI

from dhenara.ai.providers.base import AIModelProviderClientBase
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum

from .formatter import OpenAIFormatter

logger = logging.getLogger(__name__)


def _normalize_microsoft_openai_base_url(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    if not normalized:
        raise ValueError("Microsoft OpenAI endpoint must be a non-empty string")
    if normalized.endswith("/models"):
        normalized = normalized[: -len("/models")]
    if "/openai/deployments/" in normalized:
        raise ValueError(
            "Microsoft OpenAI endpoint must be the resource root or v1 base URL, not a deployment-specific path."
        )
    if normalized.endswith("/openai/v1"):
        return f"{normalized}/"
    return f"{normalized}/openai/v1/"


def map_openai_reasoning_effort(
    effort: str | None,
    supported_efforts: Sequence[str] | None = None,
) -> str | None:
    """Map Dhenara-only aliases while leaving explicit provider values unchanged."""
    if effort is None:
        return None

    value = (effort or "").lower()
    supported = tuple(item.lower() for item in supported_efforts or ())

    if value == "minimal":
        return supported[0] if supported else "low"
    if value == "max":
        return supported[-1] if supported else "high"
    return value


def map_deepseek_reasoning_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    value = effort.lower()
    if value in {"max", "xhigh"}:
        return "max"
    return "high"


# -----------------------------------------------------------------------------
class OpenAIClientBase(AIModelProviderClientBase):
    """Base class for all OpenAI Clients"""

    formatter = OpenAIFormatter

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def _get_client_params(self, api: Any) -> tuple[str, dict]:
        """Common logic for both sync and async clients"""
        http_params = self._get_client_http_params(api) or {}
        if api.provider == AIModelAPIProviderEnum.OPEN_AI:
            params = {
                "api_key": api.api_key,
                **http_params,
            }
            return "openai", params

        elif api.provider == AIModelAPIProviderEnum.MICROSOFT_OPENAI:
            client_params = api.get_provider_credentials()
            params = {
                "api_key": client_params["api_key"],
                "base_url": _normalize_microsoft_openai_base_url(client_params["base_url"]),
                **http_params,
            }
            return "openai", params

        elif api.provider == AIModelAPIProviderEnum.DEEPSEEK:
            client_params = api.get_provider_credentials()
            params = {
                "api_key": client_params["api_key"],
                "base_url": client_params["base_url"].rstrip("/"),
                **http_params,
            }
            return "openai", params

        elif api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError(
                "microsoft_azure_ai is not supported. "
                "Use microsoft_openai with an Azure OpenAI or Microsoft Foundry OpenAI v1 endpoint instead."
            )

        error_msg = f"Unsupported API provider {api.provider} for OpenAI functions"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _setup_client_sync(self):
        """Get the appropriate sync OpenAI client"""
        api = self.model_endpoint.api
        client_type, params = self._get_client_params(api)

        if client_type == "openai":
            return OpenAI(**params)

        raise ValueError(f"Unsupported OpenAI client type: {client_type}")

    async def _setup_client_async(self):
        """Get the appropriate async OpenAI client"""
        api = self.model_endpoint.api
        client_type, params = self._get_client_params(api)

        if client_type == "openai":
            return AsyncOpenAI(**params)

        raise ValueError(f"Unsupported OpenAI client type: {client_type}")
