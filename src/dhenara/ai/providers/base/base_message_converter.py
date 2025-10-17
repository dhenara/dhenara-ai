from abc import ABC, abstractmethod
from typing import Any

from dhenara.ai.types.genai import (
    ChatResponseContentItem,
)
from dhenara.ai.types.genai.ai_model import AIModelEndpoint, AIModelProviderEnum
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response import ChatResponse, ChatResponseChoice


class BaseMessageConverter(ABC):
    @staticmethod
    @abstractmethod
    def provider_message_to_dai_content_items(
        *,
        message: Any,
        structured_output_config: StructuredOutputConfig | None = None,
    ) -> list[ChatResponseContentItem]:
        pass

    @staticmethod
    @abstractmethod
    def dai_response_to_provider_message(
        dai_response: ChatResponse,
        model_endpoint: AIModelEndpoint,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        pass

    @staticmethod
    @abstractmethod
    def dai_choice_to_provider_message(
        choice: ChatResponseChoice,
        model_endpoint: AIModelEndpoint,
        source_provider: AIModelProviderEnum,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        pass
