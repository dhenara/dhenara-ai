from __future__ import annotations

from types import SimpleNamespace

import pytest

from dhenara.ai.providers.anthropic import AnthropicChat
from dhenara.ai.providers.google import GoogleAIChat
from dhenara.ai.providers.openai.responses import OpenAIResponses
from dhenara.ai.types.genai.foundation_models.anthropic.chat import ClaudeHaiku45, ClaudeSonnet46
from dhenara.ai.types.genai.foundation_models.google.chat import Gemini25Flash, Gemini31ProPreview
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT54, GPT54Mini, GPT54Nano, GPT55
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
from dhenara.ai.types.genai.dhenara.request import AIModelCallConfig, WebSearchHostedTool

pytestmark = [pytest.mark.unit]


def _make_endpoint(provider: AIModelProviderEnum, api_provider: AIModelAPIProviderEnum, model_name: str) -> AIModelEndpoint:
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


@pytest.mark.case_id("DAI-121")
def test_dai_121_anthropic_provider_side_web_search_in_chat_args():
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
def test_dai_122_google_provider_side_web_search_in_generate_config():
    ep = _make_endpoint(AIModelProviderEnum.GOOGLE_AI, AIModelAPIProviderEnum.GOOGLE_AI, "gemini-2.5-flash")
    cfg = AIModelCallConfig(
        hosted_tools=[WebSearchHostedTool()]
    )
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
def test_dai_123_openai_provider_side_web_search_usage_is_normalized():
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
def test_dai_124_anthropic_provider_side_web_search_usage_is_normalized():
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
    assert usage.hosted_tool_usage.details == {
        "provider_usage": {"server_tool_use": {"web_search_requests": 2}}
    }


@pytest.mark.case_id("DAI-125")
def test_dai_125_google_provider_side_web_search_usage_is_normalized():
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
def test_dai_126_current_foundation_models_include_provider_side_web_search_cost_rules():
    openai_models = [GPT55, GPT54, GPT54Mini, GPT54Nano]
    for model in openai_models:
        rules = model.cost_data.hosted_tool_cost_rules
        assert rules is not None
        assert len(rules) == 1
        assert rules[0].usage_key == "web_search"
        assert rules[0].flat_cost_per_unit == 0.01

    anthropic_models = [ClaudeSonnet46, ClaudeHaiku45]
    for model in anthropic_models:
        rules = model.cost_data.hosted_tool_cost_rules
        assert rules is not None
        assert len(rules) == 1
        assert rules[0].usage_key == "web_search"
        assert rules[0].flat_cost_per_unit == 0.01

    gemini31_rules = Gemini31ProPreview.cost_data.hosted_tool_cost_rules
    assert gemini31_rules is not None
    assert gemini31_rules[0].usage_key == "web_search_queries"
    assert gemini31_rules[0].flat_cost_per_unit == 0.014

    gemini25_rules = Gemini25Flash.cost_data.hosted_tool_cost_rules
    assert gemini25_rules is not None
    assert gemini25_rules[0].usage_key == "grounded_prompt"
    assert gemini25_rules[0].flat_cost_per_unit == 0.035