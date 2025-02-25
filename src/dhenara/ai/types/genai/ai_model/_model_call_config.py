from dhenara.ai.types.genai.ai_model import AIModel
from dhenara.ai.types.shared.base import BaseModel


class AIModelCallConfig(BaseModel):
    """Configuration for AI model calls"""

    max_tokens: int | None = None
    streaming: bool = False
    options: dict = None
    metadata: dict = None
    timeout: float | None = None
    retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 10.0
    test_mode: bool = False

    def get_user(self):
        user = self.metadata.get("user", None)
        if not user:
            user = self.metadata.get("user_id", None)

        return user

    def get_max_tokens(self, model: AIModel = None) -> int:
        if self.max_tokens:
            return self.max_tokens
        if not model:
            raise ValueError("Model should be passed when max_token is not set in the call-config")

        _settings = model.get_settings()
        if not _settings.max_output_tokens:
            raise ValueError(f"max_output_tokens is not set in model {model.model_name} or call-config")

        return _settings.max_output_tokens
