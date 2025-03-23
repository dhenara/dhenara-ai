from dhenara.ai.types.shared.base import BaseEnum


class AIModelFunctionalTypeEnum(BaseEnum):
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"


class AIModelProviderEnum(BaseEnum):
    CUSTOM = "custom"
    OPEN_AI = "open_ai"
    GOOGLE_AI = "google_ai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    META = "meta"
    COHERE = "cohere"


class AIModelAPIProviderEnum(BaseEnum):
    OPEN_AI = "openai"
    GOOGLE_AI = "google_gemini_api"
    ANTHROPIC = "anthropic"
    GOOGLE_VERTEX_AI = "google_vertex_ai"
    MICROSOFT_OPENAI = "microsoft_openai"
    MICROSOFT_AZURE_AI = "microsoft_azure_ai"
    AMAZON_BEDROCK = "amazon_bedrock"
    DEEPSEEK = "deepseek"
