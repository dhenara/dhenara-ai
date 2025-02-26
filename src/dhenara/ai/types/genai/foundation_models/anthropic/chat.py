from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)

Claude37Sonnet = FoundationModel(
    model_name="claude-3-7-sonnet",
    display_name="Claude Sonnet 3.7",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=8192,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=64000,
    ),
    valid_options={},
    metadata={
        "details": "Model, with highest level of intelligence and capability.",
        "version_suffix": "-latest",  # NOTE: Version is required for Anthropic API calls
    },
    order=81,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=3.0,
        output_token_cost_per_million=15.0,
    ),
)


Claude35Sonnet = FoundationModel(
    model_name="claude-3-5-sonnet",
    display_name="Claude Sonnet 3.5",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=8192,
    ),
    valid_options={},
    metadata={
        "details": "Model, with highest level of intelligence and capability.",
        "version_suffix": "-latest",  # NOTE: Version is required for Anthropic API calls
    },
    order=81,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=3.0,
        output_token_cost_per_million=15.0,
    ),
)

Claude35Haiku = FoundationModel(
    model_name="claude-3-5-haiku",
    display_name="Claude Haiku 3.5",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=8192,
    ),
    valid_options={},
    metadata={
        "details": "Fastest, most cost-effective model.",
        "version_suffix": "-latest",  # NOTE: Version is required for Anthropic API calls
    },
    order=82,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=1.0,
        output_token_cost_per_million=5.0,
    ),
)


Claude3Opus = FoundationModel(
    model_name="claude-3-opus",
    display_name="Claude 3 Opus",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=4096,
    ),
    valid_options={},
    metadata={
        "details": "Powerful model for highly complex tasks",
        "version_suffix": "-latest",  # NOTE: Version is required for Anthropic API calls
    },
    order=93,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=15.0,
        output_token_cost_per_million=75.0,
    ),
)

CHAT_MODELS = [
    Claude37Sonnet,
    Claude35Sonnet,
    Claude35Haiku,
    Claude3Opus,
]
