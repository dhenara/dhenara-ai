from __future__ import annotations

from types import SimpleNamespace

import pytest

from dhenara.ai.providers.anthropic import AnthropicChat
from dhenara.ai.providers.google import GoogleAIChat
from dhenara.ai.providers.openai.responses import OpenAIResponses
from dhenara.ai.types.genai.ai_model import (
    AIModelAPI,
    AIModelAPIProviderEnum,
    AIModelEndpoint,
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)
from dhenara.ai.types.genai.dhenara.request import (
    AIModelCallConfig,
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    ToolDefinition,
    WebSearchHostedTool,
)
from dhenara.ai.types.genai.foundation_models.anthropic.chat import (
    ClaudeHaiku45,
    ClaudeOpus45,
    ClaudeOpus46,
    ClaudeOpus47,
    ClaudeSonnet45,
    ClaudeSonnet46,
)
from dhenara.ai.types.genai.foundation_models.google.chat import (
    Gemini3FlashPreview,
    Gemini3Pro,
    Gemini25Flash,
    Gemini25FlashLite,
    Gemini25Pro,
    Gemini31FlashLitePreview,
    Gemini31ProPreview,
)
from dhenara.ai.types.genai.foundation_models.openai.chat import (
    GPT5,
    GPT51,
    GPT52,
    GPT54,
    GPT55,
    GPT5Mini,
    GPT5Nano,
    GPT51Codex,
    GPT51CodexMini,
    GPT52Pro,
    GPT54Mini,
    GPT54Nano,
    GPT54Pro,
)

pytestmark = [pytest.mark.unit]


def _make_endpoint(
    provider: AIModelProviderEnum, api_provider: AIModelAPIProviderEnum, model_name: str
) -> AIModelEndpoint:
    model = FoundationModel(
        model_name=model_name,
        display_name=model_name,
        provider=provider,
        functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
        settings=ChatModelSettings(
            max_input_tokens=100000,
            max_output_tokens=4096,
            max_output_tokens_reasoning_mode=8192,
            max_reasoning_tokens=2048,
            supports_reasoning=True,
        ),
        valid_options={},
        cost_data=ChatModelCostData(
            input_token_cost_per_million=1.0,
            output_token_cost_per_million=2.0,
        ),
    )
    api = AIModelAPI(provider=api_provider, api_key="test-key")
    return AIModelEndpoint(api=api, ai_model=model)


def _make_function_tool(name: str = "fetch_signal") -> ToolDefinition:
    params = FunctionParameters(
        properties={
            "path": FunctionParameter(type="string", required=True, description="Path to fetch"),
        },
        required=["path"],
    )
    function_def = FunctionDefinition(name=name, description="Fetch verification data", parameters=params)
    return ToolDefinition(function=function_def)


@pytest.mark.case_id("DAI-121")
def test_dai_121_anthropic_hosted_web_search_in_chat_args():
    ep = _make_endpoint(AIModelProviderEnum.ANTHROPIC, AIModelAPIProviderEnum.ANTHROPIC, "claude-sonnet-4-6")
    cfg = AIModelCallConfig(
        hosted_tools=[
            WebSearchHostedTool(
                max_uses=3,
                allowed_domains=["anthropic.com"],
            )
        ]
    )
    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})
    chat_args = params["chat_args"]
    tools = chat_args.get("tools")

    assert isinstance(tools, list)
    assert tools[0]["type"] == "web_search_20260209"
    assert tools[0]["name"] == "web_search"
    assert tools[0]["max_uses"] == 3
    assert tools[0]["allowed_domains"] == ["anthropic.com"]


@pytest.mark.case_id("DAI-122")
def test_dai_122_google_hosted_web_search_in_generate_config():
    ep = _make_endpoint(AIModelProviderEnum.GOOGLE_AI, AIModelAPIProviderEnum.GOOGLE_AI, "gemini-2.5-flash")
    cfg = AIModelCallConfig(hosted_tools=[WebSearchHostedTool()])
    client = GoogleAIChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "parts": [{"text": "hi"}]})
    generate_config = params["generate_config"]
    tools = getattr(generate_config, "tools", None)

    assert tools is not None
    assert len(tools) == 1
    assert getattr(tools[0], "google_search", None) is not None


@pytest.mark.case_id("DAI-123")
def test_dai_123_openai_hosted_web_search_usage_is_normalized():
    ep = _make_endpoint(AIModelProviderEnum.OPEN_AI, AIModelAPIProviderEnum.OPEN_AI, "gpt-5")
    client = OpenAIResponses(model_endpoint=ep, config=AIModelCallConfig(), is_async=False)

    usage = client._get_usage_from_provider_response(
        SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=45,
                total_tokens=165,
                output_tokens_details=SimpleNamespace(reasoning_tokens=11),
            ),
            output=[SimpleNamespace(type="web_search_call"), SimpleNamespace(type="message")],
        )
    )

    assert usage is not None
    assert usage.hosted_tool_usage is not None
    assert usage.hosted_tool_usage.request_counts == {"web_search": 1, "total": 1}
    assert usage.hosted_tool_usage.billing_counts == {"web_search": 1}
    assert usage.hosted_tool_usage.details == {"provider_usage": {"web_search_call_count": 1}}


@pytest.mark.case_id("DAI-124")
def test_dai_124_anthropic_hosted_web_search_usage_is_normalized():
    ep = _make_endpoint(AIModelProviderEnum.ANTHROPIC, AIModelAPIProviderEnum.ANTHROPIC, "claude-sonnet-4-6")
    client = AnthropicChat(model_endpoint=ep, config=AIModelCallConfig(), is_async=False)

    usage = client._get_usage_from_provider_response(
        SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                server_tool_use=SimpleNamespace(model_dump=lambda: {"web_search_requests": 2}),
            )
        )
    )

    assert usage.hosted_tool_usage is not None
    assert usage.hosted_tool_usage.request_counts == {"web_search": 2, "total": 2}
    assert usage.hosted_tool_usage.billing_counts == {"web_search": 2}
    assert usage.hosted_tool_usage.details == {"provider_usage": {"server_tool_use": {"web_search_requests": 2}}}


@pytest.mark.case_id("DAI-125")
def test_dai_125_google_hosted_web_search_usage_is_normalized():
    ep = _make_endpoint(AIModelProviderEnum.GOOGLE_AI, AIModelAPIProviderEnum.GOOGLE_AI, "gemini-2.5-flash")
    client = GoogleAIChat(model_endpoint=ep, config=AIModelCallConfig(), is_async=False)

    usage = client._get_usage_from_provider_response(
        SimpleNamespace(
            usage_metadata=SimpleNamespace(
                total_token_count=210,
                prompt_token_count=100,
                candidates_token_count=80,
                thoughts_token_count=20,
                tool_use_prompt_token_count=10,
            ),
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=["latest news"],
                    )
                )
            ],
        )
    )

    assert usage.hosted_tool_usage is not None
    assert usage.hosted_tool_usage.request_counts == {"web_search": 1, "total": 1}
    assert usage.hosted_tool_usage.token_counts == {"prompt": 10}
    assert usage.hosted_tool_usage.billing_counts == {"grounded_prompt": 1}
    assert usage.hosted_tool_usage.details == {
        "provider_usage": {"web_search_queries": 1, "tool_use_prompt_tokens": 10}
    }


@pytest.mark.case_id("DAI-126")
def test_dai_126_current_foundation_models_include_hosted_web_search_cost_rules():
    openai_models = [
        GPT55,
        GPT54,
        GPT54Pro,
        GPT54Mini,
        GPT54Nano,
        GPT52,
        GPT52Pro,
        GPT51,
        GPT51Codex,
        GPT51CodexMini,
        GPT5,
        GPT5Mini,
        GPT5Nano,
    ]
    for model in openai_models:
        rules = model.cost_data.hosted_tool_cost_rules
        assert rules is not None
        assert len(rules) == 1
        assert rules[0].usage_key == "web_search"
        assert rules[0].flat_cost_per_unit == 0.01

    anthropic_models = [ClaudeOpus47, ClaudeOpus46, ClaudeSonnet46, ClaudeOpus45, ClaudeSonnet45, ClaudeHaiku45]
    for model in anthropic_models:
        rules = model.cost_data.hosted_tool_cost_rules
        assert rules is not None
        assert len(rules) == 1
        assert rules[0].usage_key == "web_search"
        assert rules[0].flat_cost_per_unit == 0.01

    gemini3_models = [Gemini3Pro, Gemini31ProPreview, Gemini3FlashPreview, Gemini31FlashLitePreview]
    for model in gemini3_models:
        rules = model.cost_data.hosted_tool_cost_rules
        assert rules is not None
        assert len(rules) == 1
        assert rules[0].usage_key == "web_search_queries"
        assert rules[0].flat_cost_per_unit == 0.014

    gemini25_models = [Gemini25Pro, Gemini25Flash, Gemini25FlashLite]
    for model in gemini25_models:
        rules = model.cost_data.hosted_tool_cost_rules
        assert rules is not None
        assert len(rules) == 1
        assert rules[0].usage_key == "grounded_prompt"
        assert rules[0].flat_cost_per_unit == 0.035


@pytest.mark.case_id("DAI-127")
def test_dai_127_openai_hosted_tool_format_failure_keeps_function_tools(caplog):
    ep = _make_endpoint(AIModelProviderEnum.OPEN_AI, AIModelAPIProviderEnum.OPEN_AI, "gpt-5.4")
    cfg = AIModelCallConfig(
        tools=[_make_function_tool()],
        hosted_tools=[WebSearchHostedTool(max_uses=1)],
    )
    client = OpenAIResponses(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    with caplog.at_level("ERROR"):
        params = client.get_api_call_params(prompt=None, context=None, instructions=None, messages=None)

    tools = params["response_args"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "fetch_signal"
    assert "continuing without them" in caplog.text


@pytest.mark.case_id("DAI-128")
def test_dai_128_google_hosted_tool_format_failure_keeps_function_tools(caplog):
    ep = _make_endpoint(AIModelProviderEnum.GOOGLE_AI, AIModelAPIProviderEnum.GOOGLE_AI, "gemini-3.1-pro-preview")
    cfg = AIModelCallConfig(
        tools=[_make_function_tool()],
        hosted_tools=[WebSearchHostedTool(allowed_domains=["example.com"])],
    )
    client = GoogleAIChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    with caplog.at_level("ERROR"):
        params = client.get_api_call_params(prompt={"role": "user", "parts": [{"text": "hi"}]})

    tools = getattr(params["generate_config"], "tools", None)
    assert tools is not None
    assert len(tools) == 1
    assert getattr(tools[0], "function_declarations", None) is not None
    assert "continuing without them" in caplog.text or "continuing without it" in caplog.text


@pytest.mark.case_id("DAI-129")
def test_dai_129_anthropic_hosted_tool_format_failure_keeps_function_tools(caplog):
    ep = _make_endpoint(AIModelProviderEnum.ANTHROPIC, AIModelAPIProviderEnum.ANTHROPIC, "claude-sonnet-4-6")
    cfg = AIModelCallConfig(
        tools=[_make_function_tool()],
        hosted_tools=[WebSearchHostedTool(search_context_size="medium")],
    )
    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    with caplog.at_level("ERROR"):
        params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})

    tools = params["chat_args"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "fetch_signal"
    assert "continuing without them" in caplog.text


@pytest.mark.case_id("DAI-130")
def test_dai_130_openai_hosted_tool_usage_failure_preserves_token_usage(caplog):
    class BadOutput:
        def __iter__(self):
            raise RuntimeError("boom")

    ep = _make_endpoint(AIModelProviderEnum.OPEN_AI, AIModelAPIProviderEnum.OPEN_AI, "gpt-5")
    client = OpenAIResponses(model_endpoint=ep, config=AIModelCallConfig(), is_async=False)

    with caplog.at_level("ERROR"):
        usage = client._get_usage_from_provider_response(
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=120,
                    output_tokens=45,
                    total_tokens=165,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=11),
                ),
                output=BadOutput(),
            )
        )

    assert usage is not None
    assert usage.total_tokens == 165
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 45
    assert usage.reasoning_tokens == 11
    assert usage.hosted_tool_usage is None
    assert "continuing without it" in caplog.text


@pytest.mark.case_id("DAI-131")
def test_dai_131_google_hosted_tool_usage_failure_preserves_token_usage(caplog):
    class BadCandidate:
        @property
        def grounding_metadata(self):
            raise RuntimeError("boom")

    ep = _make_endpoint(AIModelProviderEnum.GOOGLE_AI, AIModelAPIProviderEnum.GOOGLE_AI, "gemini-3.1-pro-preview")
    client = GoogleAIChat(model_endpoint=ep, config=AIModelCallConfig(), is_async=False)

    with caplog.at_level("ERROR"):
        usage = client._get_usage_from_provider_response(
            SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    total_token_count=210,
                    prompt_token_count=100,
                    candidates_token_count=80,
                    thoughts_token_count=20,
                    tool_use_prompt_token_count=10,
                ),
                candidates=[BadCandidate()],
            )
        )

    assert usage.total_tokens == 210
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 110
    assert usage.reasoning_tokens == 20
    assert usage.hosted_tool_usage is None
    assert "continuing without it" in caplog.text


@pytest.mark.case_id("DAI-132")
def test_dai_132_anthropic_hosted_tool_usage_failure_preserves_token_usage(caplog):
    class BadServerToolUse:
        def model_dump(self):
            raise RuntimeError("boom")

    ep = _make_endpoint(AIModelProviderEnum.ANTHROPIC, AIModelAPIProviderEnum.ANTHROPIC, "claude-sonnet-4-6")
    client = AnthropicChat(model_endpoint=ep, config=AIModelCallConfig(), is_async=False)

    with caplog.at_level("ERROR"):
        usage = client._get_usage_from_provider_response(
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    server_tool_use=BadServerToolUse(),
                )
            )
        )

    assert usage.total_tokens == 150
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.hosted_tool_usage is None
    assert "continuing without it" in caplog.text
