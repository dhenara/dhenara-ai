from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)

Gemini15Flash = FoundationModel(
    model_name="gemini-1.5-flash",
    display_name="Gemini 1.5 Flash",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_input_tokens=1048576,
        max_output_tokens=8192,
    ),
    valid_options={},
    metadata={
        "details": "GoogleAI gemini-1.5-flash model, Optimized for Fast and versatile performance across a diverse variety of tasks",
        "display_order": 10,
    },
    order=20,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.15,
        output_token_cost_per_million=0.60,
    ),
)

Gemini15Pro = FoundationModel(
    model_name="gemini-1.5-pro",
    display_name="Gemini 1.5 Pro",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_input_tokens=2097152,
        max_output_tokens=8192,
    ),
    valid_options={},
    metadata={
        "details": "GoogleAI gemini-1.5-pro model, Optimized for complex reasoning tasks",
        "display_order": 11,
    },
    order=21,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=2.50,
        output_token_cost_per_million=10.0,
    ),
)

CHAT_MODELS = [Gemini15Flash, Gemini15Pro]
