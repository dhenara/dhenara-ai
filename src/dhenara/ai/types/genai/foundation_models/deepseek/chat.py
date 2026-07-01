from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)

DeepseekV4Flash = FoundationModel(
    model_name="deepseek-v4-flash",
    display_name="DeepSeek V4 Flash",
    provider=AIModelProviderEnum.DEEPSEEK,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=384000,
        supports_reasoning=True,
        max_output_tokens_reasoning_mode=384000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={
        "details": "DeepSeek V4 Flash model with thinking and non-thinking modes.",
    },
    order=20,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.14,
        output_token_cost_per_million=0.28,
    ),
)

DeepseekV4Pro = FoundationModel(
    model_name="deepseek-v4-pro",
    display_name="DeepSeek V4 Pro",
    provider=AIModelProviderEnum.DEEPSEEK,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=384000,
        supports_reasoning=True,
        max_output_tokens_reasoning_mode=384000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={
        "details": "DeepSeek V4 Pro model with thinking and non-thinking modes.",
    },
    order=21,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.435,
        output_token_cost_per_million=0.87,
    ),
)

DeepseekR1 = FoundationModel(
    model_name="DeepSeek-R1",
    display_name="DeepSeek-R1",
    provider=AIModelProviderEnum.DEEPSEEK,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    enabled=False,
    settings=ChatModelSettings(
        max_context_window_tokens=128000,
        max_output_tokens=8000,
    ),
    valid_options={},
    metadata={
        "deprecated": True,
        "details": "Deprecated local catalog entry. Use deepseek-v4-flash or deepseek-v4-pro for API calls.",
    },
    order=90,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.60,
        output_token_cost_per_million=2.20,
    ),
)

CHAT_MODELS = [DeepseekV4Flash, DeepseekV4Pro]
