from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)

GPT4o = FoundationModel(
    model_name="gpt-4o",
    display_name="GPT-4o",
    provider=AIModelProviderEnum.OPEN_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=128000,
        max_output_tokens=16384,
    ),
    valid_options={},
    metadata={
        "details": "OpenAI GPT-4o model, optimized for conversational AI.",
    },
    order=10,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=2.5,
        output_token_cost_per_million=10.0,
    ),
)


GPT4oMini = FoundationModel(
    model_name="gpt-4o-mini",
    display_name="GPT-4o-mini",
    provider=AIModelProviderEnum.OPEN_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=128000,
        max_output_tokens=16384,
    ),
    valid_options={},
    metadata={
        "details": "OpenAI's affordable and intelligent small model for fast, lightweight tasks.",
    },
    order=11,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=0.15,
        output_token_cost_per_million=0.60,
    ),
)

O1 = FoundationModel(
    model_name="o1",
    display_name="o1",
    provider=AIModelProviderEnum.OPEN_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=100000,
    ),
    valid_options={},
    metadata={
        "details": "OpenAI o1 model, optimized for reasoning.",
    },
    order=20,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=15.0,
        output_token_cost_per_million=60.0,
    ),
)

O1Mini = FoundationModel(
    model_name="o1-mini",
    display_name="o1-mini",
    provider=AIModelProviderEnum.OPEN_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=128000,
        max_output_tokens=65536,
    ),
    valid_options={},
    metadata={
        "details": "OpenAI o1-mini model, optimized for reasoning.",
        "display_order": 20,
    },
    order=21,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=1.10,
        output_token_cost_per_million=4.40,
    ),
)

O3Mini = FoundationModel(
    model_name="o3-mini",
    display_name="o3-mini",
    provider=AIModelProviderEnum.OPEN_AI,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=100000,
    ),
    valid_options={},
    metadata={
        "details": "OpenAI o3-mini model, optimized for reasoning.",
    },
    order=22,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=1.10,
        output_token_cost_per_million=4.40,
    ),
)

CHAT_MODELS = [GPT4o, GPT4oMini, O1, O1Mini, O3Mini]
