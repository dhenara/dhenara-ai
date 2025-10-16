from abc import ABC, abstractmethod
from typing import Any

from dhenara.ai.types.genai import (
    ChatResponseContentItem,
)
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response import ChatResponseChoice


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
    def choice_to_provider_messages(
        choice: ChatResponseChoice,
        *,
        model: str | None = None,
        provider: str | None = None,
        strict_same_provider: bool = False,
    ) -> dict[str, object]:
        pass
