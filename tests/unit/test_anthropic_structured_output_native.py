from __future__ import annotations

import pytest
from pydantic import BaseModel

from dhenara.ai.providers.anthropic.chat import AnthropicChat
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.genai.ai_model import AIModelAPI, AIModelAPIProviderEnum
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude37Sonnet, ClaudeOpus46, ClaudeSonnet45

pytestmark = [pytest.mark.unit]


class TravelPlan(BaseModel):
    destination: str
    days: int
    interests: list[str]


def _mk_ep(model) -> AIModelEndpoint:
    api = AIModelAPI(
        provider=AIModelAPIProviderEnum.ANTHROPIC,
        api_key="sk-testkey-123456",
    )
    return AIModelEndpoint(api=api, ai_model=model)


@pytest.mark.case_id("DAI-112")
def test_dai_112_anthropic_native_structured_output_uses_output_config_for_45_models():
    ep = _mk_ep(ClaudeSonnet45)
    cfg = AIModelCallConfig(structured_output=TravelPlan)

    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    # Allow calling get_api_call_params() without context manager.
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})
    chat_args = params["chat_args"]

    assert "output_config" in chat_args
    assert chat_args["output_config"]["format"]["type"] == "json_schema"
    assert "schema" in chat_args["output_config"]["format"]

    # Native structured output should not inject a structured-output tool.
    assert "tools" not in chat_args or chat_args["tools"] in (None, [])


@pytest.mark.case_id("DAI-114")
def test_dai_114_anthropic_opus46_uses_adaptive_thinking_and_output_config_effort():
    ep = _mk_ep(ClaudeOpus46)
    cfg = AIModelCallConfig(
        structured_output=TravelPlan,
        reasoning=True,
        reasoning_effort="medium",
    )

    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})
    chat_args = params["chat_args"]

    # Opus 4.6: adaptive thinking (budget_tokens is deprecated)
    assert chat_args.get("thinking") == {"type": "adaptive"}

    # Effort is configured via output_config on Opus 4.6.
    assert "output_config" in chat_args
    assert chat_args["output_config"]["effort"] == "medium"

    # Structured output should still be native on Opus 4.6.
    assert chat_args["output_config"]["format"]["type"] == "json_schema"
    assert "schema" in chat_args["output_config"]["format"]

    # Native structured output should not inject a structured-output tool.
    assert "tools" not in chat_args or chat_args["tools"] in (None, [])


@pytest.mark.case_id("DAI-113")
def test_dai_113_anthropic_structured_output_falls_back_to_tool_mode_for_37_models():
    ep = _mk_ep(Claude37Sonnet)
    cfg = AIModelCallConfig(structured_output=TravelPlan)

    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})
    chat_args = params["chat_args"]

    assert "output_config" not in chat_args
    assert "tools" in chat_args
    assert any(t.get("name") in ("TravelPlan", "structured_output") for t in chat_args["tools"])

    # In tool mode (without thinking), we enforce that tool.
    assert "tool_choice" in chat_args
    assert chat_args["tool_choice"]["type"] in ("tool", "auto")
