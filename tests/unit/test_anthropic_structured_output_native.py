from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from dhenara.ai.providers.anthropic.chat import AnthropicChat
from dhenara.ai.providers.anthropic.formatter import (
    AnthropicFormatter,
    AnthropicNativeStructuredOutputSchemaTooLargeError,
)
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.genai.ai_model import AIModelAPI, AIModelAPIProviderEnum
from dhenara.ai.types.genai.dhenara.request import (
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    ToolChoice,
    ToolDefinition,
)
from dhenara.ai.types.genai.foundation_models.anthropic.chat import (
    Claude37Sonnet,
    ClaudeOpus46,
    ClaudeOpus47,
    ClaudeSonnet45,
)

pytestmark = [pytest.mark.unit]


class TravelPlan(BaseModel):
    destination: str
    days: int
    interests: list[str]


class VerboseChild(BaseModel):
    primary: str = Field(description="Primary nested value with intentionally verbose schema metadata.")
    secondary: str | None = Field(
        default=None,
        description="Secondary optional nested value with intentionally verbose schema metadata.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Nested note rows with intentionally verbose schema metadata.",
    )


class VerboseEnvelope(BaseModel):
    child_a: VerboseChild = Field(description="First nested child object with verbose descriptive metadata.")
    child_b: VerboseChild | None = Field(
        default=None,
        description="Second optional nested child object with verbose descriptive metadata.",
    )
    child_rows: list[VerboseChild] = Field(description="Repeated child rows with verbose descriptive metadata.")
    summary: str | None = Field(default=None, description="Optional prose summary with verbose metadata.")
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="List of gaps with verbose metadata.",
    )


def _mk_ep(model) -> AIModelEndpoint:
    api = AIModelAPI(
        provider=AIModelAPIProviderEnum.ANTHROPIC,
        api_key="sk-testkey-123456",
    )
    return AIModelEndpoint(api=api, ai_model=model)


def _mk_tool(name: str = "fetch_signal") -> ToolDefinition:
    params = FunctionParameters(
        properties={
            "path": FunctionParameter(type="string", required=True, description="Path to inspect"),
        },
        required=["path"],
    )
    return ToolDefinition(
        function=FunctionDefinition(name=name, description="Fetch verification data", parameters=params)
    )


def _contains_schema_metadata(value) -> bool:
    if isinstance(value, dict):
        if any(key in value for key in ("title", "description", "default")):
            return True
        return any(_contains_schema_metadata(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_schema_metadata(item) for item in value)
    return False


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


@pytest.mark.case_id("DAI-305")
def test_dai_305_anthropic_native_schema_strips_non_contract_metadata():
    ep = _mk_ep(ClaudeOpus47)
    cfg = AIModelCallConfig(structured_output=VerboseEnvelope)

    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})
    chat_args = params["chat_args"]
    schema = chat_args["output_config"]["format"]["schema"]

    assert chat_args["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in chat_args or chat_args["tools"] in (None, [])
    assert not _contains_schema_metadata(schema)
    assert schema["properties"]["child_a"]["$ref"] == "#/$defs/VerboseChild"
    assert sorted(schema["required"]) == sorted(schema["properties"].keys())


@pytest.mark.case_id("DAI-306")
def test_dai_306_anthropic_native_schema_size_fallback_logs_error(caplog, monkeypatch):
    ep = _mk_ep(ClaudeSonnet45)
    cfg = AIModelCallConfig(structured_output=TravelPlan)

    def fake_output_config(_cls, *, structured_output, model_endpoint=None):
        raise AnthropicNativeStructuredOutputSchemaTooLargeError(
            "Anthropic native json_schema too large (5001 bytes > 4000 bytes)"
        )

    monkeypatch.setattr(
        AnthropicFormatter,
        "format_structured_output_output_config",
        classmethod(fake_output_config),
    )

    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    with caplog.at_level("ERROR"):
        params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})

    chat_args = params["chat_args"]
    assert "output_config" not in chat_args
    assert any(t.get("name") in ("TravelPlan", "structured_output") for t in chat_args["tools"])
    assert "fallback to tool mode" in caplog.text
    assert "5001 bytes > 4000 bytes" in caplog.text


@pytest.mark.case_id("DAI-304")
def test_dai_304_anthropic_formatter_preserves_boolean_type_and_date_format_for_tools():
    params = FunctionParameters(
        properties={
            "has_attachment": FunctionParameter(type="boolean", description="Attachment filter"),
            "after_date": FunctionParameter(type="string", format="date", description="Lower date bound"),
        }
    )

    converted = AnthropicChat.formatter.convert_function_definition(
        FunctionDefinition(name="gmail_workspace_search", description="Search Gmail", parameters=params)
    )
    properties = converted["input_schema"]["properties"]

    assert properties["has_attachment"]["type"] == "boolean"
    assert properties["after_date"]["type"] == "string"
    assert properties["after_date"]["format"] == "date"


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


@pytest.mark.case_id("DAI-116")
def test_dai_116_anthropic_opus47_uses_adaptive_thinking_and_output_config_effort():
    ep = _mk_ep(ClaudeOpus47)
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

    assert chat_args.get("thinking") == {"type": "adaptive"}
    assert "output_config" in chat_args
    assert chat_args["output_config"]["effort"] == "medium"
    assert chat_args["output_config"]["format"]["type"] == "json_schema"
    assert "schema" in chat_args["output_config"]["format"]
    assert "tools" not in chat_args or chat_args["tools"] in (None, [])


@pytest.mark.case_id("DAI-117")
def test_dai_117_anthropic_opus47_relaxes_forced_tool_choice_when_thinking_adds_output_config():
    ep = _mk_ep(ClaudeOpus47)
    cfg = AIModelCallConfig(
        tools=[_mk_tool()],
        tool_choice=ToolChoice(type="one_or_more"),
        reasoning=True,
        reasoning_effort="medium",
    )

    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})
    chat_args = params["chat_args"]

    assert chat_args.get("thinking") == {"type": "adaptive"}
    assert chat_args["output_config"]["effort"] == "medium"
    assert chat_args["tool_choice"] == {"type": "auto"}
    assert chat_args["tools"][0]["name"] == "fetch_signal"


@pytest.mark.case_id("DAI-118")
def test_dai_118_anthropic_opus47_keeps_native_structured_output_with_tools():
    ep = _mk_ep(ClaudeOpus47)
    cfg = AIModelCallConfig(
        structured_output=TravelPlan,
        tools=[_mk_tool()],
        tool_choice=ToolChoice(type="one_or_more"),
        reasoning=True,
        reasoning_effort="medium",
    )

    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})
    chat_args = params["chat_args"]

    assert chat_args.get("thinking") == {"type": "adaptive"}
    assert chat_args["output_config"]["effort"] == "medium"
    assert chat_args["output_config"]["format"]["type"] == "json_schema"
    assert "schema" in chat_args["output_config"]["format"]
    assert chat_args["tool_choice"] == {"type": "auto"}
    assert chat_args["tools"][0]["name"] == "fetch_signal"


@pytest.mark.case_id("DAI-119")
def test_dai_119_anthropic_opus47_keeps_forced_tool_choice_without_thinking():
    ep = _mk_ep(ClaudeOpus47)
    cfg = AIModelCallConfig(
        tools=[_mk_tool()],
        tool_choice=ToolChoice(type="one_or_more"),
        reasoning=False,
    )

    client = AnthropicChat(model_endpoint=ep, config=cfg, is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})
    chat_args = params["chat_args"]

    assert "thinking" not in chat_args
    assert chat_args["tool_choice"]["type"] in ("any", "tool")
    assert chat_args["tool_choice"] != {"type": "auto"}


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
