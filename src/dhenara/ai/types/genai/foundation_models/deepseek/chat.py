from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)

DeepseekR1 = FoundationModel(
    model_name="DeepSeek-R1",
    display_name="DeepSeek-R1",
    provider=AIModelProviderEnum.DEEPSEEK,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=128000,
        max_output_tokens=8000,
    ),
    valid_options={},
    metadata={
        "details": "DeepSeek-R1 model, optimized for reasoning.",
    },
    order=20,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.60,
        output_token_cost_per_million=2.20,
    ),
)

CHAT_MODELS = [DeepseekR1]
