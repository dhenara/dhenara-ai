from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
    HostedToolCostRule,
)


def _anthropic_web_search_cost_rules() -> list[HostedToolCostRule]:
    return [
        HostedToolCostRule(
            key="hosted_tool:web_search",
            usage_bucket="billing_counts",
            usage_key="web_search",
            flat_cost_per_unit=0.01,
            unit="search",
            description="Anthropic web-search list price per search.",
        )
    ]


ClaudeFable5 = FoundationModel(
    model_name="claude-fable-5",
    display_name="Claude Fable 5",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=128000,
        supports_reasoning=True,
        max_output_tokens_reasoning_mode=128000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={
        "details": (
            "Anthropic's most capable widely released model for demanding reasoning and long-horizon agentic work."
        ),
    },
    order=60,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=10.0,
        output_token_cost_per_million=50.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)


ClaudeMythos5 = FoundationModel(
    model_name="claude-mythos-5",
    display_name="Claude Mythos 5",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    enabled=False,
    beta=True,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=128000,
        supports_reasoning=True,
        max_output_tokens_reasoning_mode=128000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={
        "details": ("Limited-availability Project Glasswing model sharing Claude Fable 5 capabilities."),
        "limited_availability": True,
    },
    order=61,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=10.0,
        output_token_cost_per_million=50.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)


ClaudeOpus48 = FoundationModel(
    model_name="claude-opus-4-8",
    display_name="Claude Opus 4.8",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=128000,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=128000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={
        "details": "Anthropic's latest Opus model for complex reasoning, agentic coding, and knowledge work.",
    },
    order=77,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=5.0,
        output_token_cost_per_million=25.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)


ClaudeOpus47 = FoundationModel(
    model_name="claude-opus-4-7",
    display_name="Claude Opus 4.7",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=128000,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=128000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={
        "details": "Anthropic's latest generally available model for complex reasoning and agentic coding.",
    },
    order=76,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=5.0,
        output_token_cost_per_million=25.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)

ClaudeOpus46 = FoundationModel(
    model_name="claude-opus-4-6",
    display_name="Claude Opus 4.6",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=128000,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=128000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={},
    order=75,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=5.0,
        output_token_cost_per_million=25.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)

ClaudeSonnet5 = FoundationModel(
    model_name="claude-sonnet-5",
    display_name="Claude Sonnet 5",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=128000,
        supports_reasoning=True,
        max_output_tokens_reasoning_mode=128000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={
        "details": "Anthropic's current Sonnet model with strong speed, intelligence, and adaptive thinking.",
    },
    order=68,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=3.0,
        output_token_cost_per_million=15.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)

ClaudeSonnet46 = FoundationModel(
    model_name="claude-sonnet-4-6",
    display_name="Claude Sonnet 4.6",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=1000000,
        max_output_tokens=64000,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=64000,
        reasoning_control="effort",
    ),
    valid_options={},
    metadata={},
    order=70,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=3.0,
        output_token_cost_per_million=15.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)

ClaudeOpus45 = FoundationModel(
    model_name="claude-opus-4-5",
    display_name="Claude Opus 4.5",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=64000,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=64000,
        reasoning_control="token_budget",
    ),
    valid_options={},
    metadata={},
    order=75,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=5.0,
        output_token_cost_per_million=25.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)


ClaudeSonnet45 = FoundationModel(
    model_name="claude-sonnet-4-5",
    display_name="Claude Sonnet 4.5",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=64000,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=64000,
        reasoning_control="token_budget",
    ),
    valid_options={},
    metadata={},
    order=70,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=3.0,
        output_token_cost_per_million=15.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)


ClaudeHaiku45 = FoundationModel(
    model_name="claude-haiku-4-5",
    display_name="Claude Haiku 4.5",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=64000,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=64000,
        reasoning_control="token_budget",
    ),
    valid_options={},
    metadata={},
    order=82,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=1.0,
        output_token_cost_per_million=5.0,
        hosted_tool_cost_rules=_anthropic_web_search_cost_rules(),
    ),
)


ClaudeSonnet40 = FoundationModel(
    model_name="claude-sonnet-4-0",
    display_name="Claude Sonnet 4",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=8192,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=64000,
        reasoning_control="token_budget",
    ),
    valid_options={},
    metadata={
        # "version_suffix": "-latest",
    },
    order=70,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=3.0,
        output_token_cost_per_million=15.0,
    ),
)

ClaudeOpus40 = FoundationModel(
    model_name="claude-opus-4-0",
    display_name="Claude Opus 4",
    provider=AIModelProviderEnum.ANTHROPIC,
    functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
    settings=ChatModelSettings(
        max_context_window_tokens=200000,
        max_output_tokens=8192,
        supports_reasoning=True,
        max_reasoning_tokens=32000,
        max_output_tokens_reasoning_mode=32000,
        reasoning_control="token_budget",
    ),
    valid_options={},
    metadata={},
    order=75,
    cost_data=ChatModelCostData(
        input_token_cost_per_million=15.0,
        output_token_cost_per_million=75.0,
    ),
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
        reasoning_control="token_budget",
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

Claude40Sonnet = ClaudeSonnet40
CHAT_MODELS = [
    ClaudeFable5,
    ClaudeMythos5,
    ClaudeOpus48,
    ClaudeOpus47,
    ClaudeOpus46,
    ClaudeSonnet5,
    ClaudeSonnet46,
    ClaudeOpus45,
    ClaudeSonnet45,
    ClaudeHaiku45,
    ClaudeOpus40,
    ClaudeSonnet40,
    Claude40Sonnet,
    Claude37Sonnet,
    Claude35Sonnet,
    Claude35Haiku,
    Claude3Opus,
]
