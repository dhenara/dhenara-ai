from dhenara.ai.types.genai.ai_model import AIModelFunctionalTypeEnum, AIModelProviderEnum

from .anthropic.chat import CHAT_MODELS as ANTHROPIC_CHAT_MODELS
from .deepseek.chat import CHAT_MODELS as DEEPSEEK_CHAT_MODELS
from .google.chat import CHAT_MODELS as GOOGLE_CHAT_MODELS
from .google.image import IMAGE_MODELS as GOOGLE_IMAGE_MODELS
from .openai.chat import CHAT_MODELS as OPENAI_CHAT_MODELS
from .openai.image import IMAGE_MODELS as OPENAI_IMAGE_MODELS

FOUNDATION_MODELS_MAPPINGS = {
    AIModelFunctionalTypeEnum.TEXT_GENERATION: {
        AIModelProviderEnum.OPEN_AI: OPENAI_CHAT_MODELS,
        AIModelProviderEnum.GOOGLE_AI: GOOGLE_CHAT_MODELS,
        AIModelProviderEnum.ANTHROPIC: ANTHROPIC_CHAT_MODELS,
        AIModelProviderEnum.DEEPSEEK: DEEPSEEK_CHAT_MODELS,
    },
    AIModelFunctionalTypeEnum.IMAGE_GENERATION: {
        AIModelProviderEnum.OPEN_AI: OPENAI_IMAGE_MODELS,
        AIModelProviderEnum.GOOGLE_AI: GOOGLE_IMAGE_MODELS,
    },
}

# fmt: off
ALL_CHAT_MODELS = [
    model
    for provider_models in FOUNDATION_MODELS_MAPPINGS[AIModelFunctionalTypeEnum.TEXT_GENERATION].values()
    for model in provider_models
]

ALL_IMAGE_MODELS = [
    model
    for provider_models in FOUNDATION_MODELS_MAPPINGS[AIModelFunctionalTypeEnum.IMAGE_GENERATION].values()
    for model in provider_models
]

ALL_FOUNDATION_MODELS= [
    model
    for functional_type in FOUNDATION_MODELS_MAPPINGS.values()
    for provider_models in functional_type.values()
    for model in provider_models
]
# fmt: on
