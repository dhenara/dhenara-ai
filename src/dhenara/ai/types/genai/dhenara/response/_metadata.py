from dhenara.ai.types.shared.base import BaseModel


class AIModelCallResponseMetaData(BaseModel):
    streaming: bool = False
    duration_seconds: int | float | None = None
    provider_metadata: dict | None = None
