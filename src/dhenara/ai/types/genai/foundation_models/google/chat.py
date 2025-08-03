from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)

Gemini25Pro = FoundationModel(
    model_name="gemini-2.5-pro",
    display_name="Gemini 2.5 Pro",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_input_tokens=1048576,
        max_output_tokens=65535,
        supports_reasoning=True,
        max_reasoning_tokens=32768,
        max_output_tokens_reasoning_mode=64000,  # TODO: Review this simple workaround when google api has more control
    ),
    valid_options={},
    metadata={
        "details": "GoogleAI gemini-2.5 Pro model",
        "display_order": 10,
    },
    order=52,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=1.25,
        output_token_cost_per_million=10.0,
    ),
)

Gemini25Flash = FoundationModel(
    model_name="gemini-2.5-flash",
    display_name="Gemini 2.5 Flash",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_input_tokens=1048576,
        max_output_tokens=65535,
        supports_reasoning=True,
        max_reasoning_tokens=32768,
        max_output_tokens_reasoning_mode=64000,  # TODO: Review this simple workaround when google api has more control
    ),
    valid_options={},
    metadata={
        "details": "GoogleAI gemini-2.5-flash model",
        "display_order": 10,
    },
    order=53,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.15,
        output_token_cost_per_million=2.50,
    ),
)


Gemini25FlashLite = FoundationModel(
    model_name="gemini-2.5-flash-lite",
    display_name="Gemini 2.5 Flash Lite",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_input_tokens=1048576,
        max_output_tokens=65535,
        supports_reasoning=True,
        max_reasoning_tokens=24576,
        max_output_tokens_reasoning_mode=64000,  # TODO: Review this simple workaround when google api has more control
    ),
    valid_options={},
    metadata={
        "details": "GoogleAI gemini-2.5-flash model",
        "display_order": 10,
    },
    order=53,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.1,
        output_token_cost_per_million=0.4,
    ),
)

Gemini20Flash = FoundationModel(
    model_name="gemini-2.0-flash",
    display_name="Gemini 2 Flash",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_input_tokens=1048576,
        max_output_tokens=8192,
    ),
    valid_options={},
    metadata={
        "details": "GoogleAI gemini-2.0-flash model",
        "display_order": 10,
    },
    order=82,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.10,
        output_token_cost_per_million=0.40,
    ),
)

Gemini20FlashLite = FoundationModel(
    model_name="gemini-2.0-flash-lite",
    display_name="Gemini 2 Flash Lite",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_input_tokens=1048576,
        max_output_tokens=8192,
    ),
    valid_options={},
    metadata={
        "details": "GoogleAI gemini-2.0-flash-light model",
        "display_order": 10,
    },
    order=83,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.075,
        output_token_cost_per_million=0.30,
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
        "display_order": 91,
    },
    order=21,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=2.50,
        output_token_cost_per_million=10.0,
    ),
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
        "details": "GoogleAI gemini-1.5-flash model",
        "display_order": 92,
    },
    order=20,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.15,
        output_token_cost_per_million=0.60,
    ),
)

CHAT_MODELS = [
    Gemini25Pro,
    Gemini25Flash,
    Gemini25FlashLite,
    Gemini20Flash,
    Gemini20FlashLite,
    Gemini15Flash,
    Gemini15Pro,
]
