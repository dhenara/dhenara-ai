from dhenara.ai.providers.anthropic import AnthropicChat
from dhenara.ai.providers.base import AIModelCallConfig, AIModelProviderClientBase
from dhenara.ai.providers.google import GoogleAIChat, GoogleAIImage
from dhenara.ai.providers.openai import OpenAIChat, OpenAIImage
from dhenara.ai.types.external_api import AIModelProviderEnum
from dhenara.ai.types.genai.ai_model import AIModelEndpoint, AIModelFunctionalTypeEnum


class AIModelClientFactory:
    """Factory for creating AI model providers"""

    _provider_clients = {
        AIModelFunctionalTypeEnum.TEXT_GENERATION: {
            AIModelProviderEnum.OPEN_AI: OpenAIChat,
            AIModelProviderEnum.ANTHROPIC: AnthropicChat,
            AIModelProviderEnum.GOOGLE_AI: GoogleAIChat,
            AIModelProviderEnum.DEEPSEEK: OpenAIChat,
        },
        AIModelFunctionalTypeEnum.IMAGE_GENERATION: {
            AIModelProviderEnum.OPEN_AI: OpenAIImage,
            AIModelProviderEnum.GOOGLE_AI: GoogleAIImage,
        },
    }

    @classmethod
    def create_provider_client(
        cls,
        model_endpoint: AIModelEndpoint,
        config: AIModelCallConfig,
        is_async: bool,
    ) -> AIModelProviderClientBase:
        functional_type_mapping = cls._provider_clients.get(model_endpoint.ai_model.functional_type)
        if not functional_type_mapping:
            raise ValueError(f"Unsupported functional_type: {model_endpoint.ai_model.functional_type}")

        provider_class = functional_type_mapping.get(model_endpoint.ai_model.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {model_endpoint.ai_model.provider}")
        return provider_class(
            model_endpoint=model_endpoint,
            config=config,
            is_async=is_async,
        )
