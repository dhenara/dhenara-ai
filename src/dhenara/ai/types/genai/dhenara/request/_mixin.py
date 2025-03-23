from typing import Any

from dhenara.ai.types.genai.ai_model import AIModelProviderEnum


class ToProviderMixin:
    def to_provider_format(self, provider: AIModelProviderEnum) -> dict[str, Any]:
        """Convert to provider format"""
        if provider == AIModelProviderEnum.OPEN_AI:
            return self.to_openai_format()
        elif provider == AIModelProviderEnum.ANTHROPIC:
            return self.to_anthropic_format()
        elif provider == AIModelProviderEnum.GOOGLE_AI:
            return self.to_google_format()
        else:
            raise ValueError(f"Provider {provider} not supported")


class FromProviderMixin:
    def from_provider_format(self, provider: AIModelProviderEnum) -> dict[str, Any]:
        """Convert to provider format"""
        if provider == AIModelProviderEnum.OPEN_AI:
            return self.from_openai_format()
        elif provider == AIModelProviderEnum.ANTHROPIC:
            return self.from_anthropic_format()
        elif provider == AIModelProviderEnum.GOOGLE_AI:
            return self.from_google_format()
        else:
            raise ValueError(f"Provider {provider} not supported")
