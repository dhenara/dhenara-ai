# Provenance: Added to improve OpenAI formatter/message converter coverage (2026-01-21)

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from dhenara.ai.providers.openai.formatter import OpenAIFormatter
from dhenara.ai.providers.openai.message_converter import OpenAIMessageConverter
from dhenara.ai.types.genai.ai_model import AIModelProviderEnum
from dhenara.ai.types.genai.dhenara.request import (
    FunctionParameter,
    FunctionParameters,
    StructuredOutputConfig,
)

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-056")
def test_dai_056_openai_formatter_cleans_schema_for_strict_mode():
    """GIVEN a JSON schema containing non-strict fields and nested object branches
    WHEN OpenAIFormatter._clean_schema_for_openai_strict_mode is called
    THEN it enforces additionalProperties=false, required=properties, and strips numeric constraints.
    """

    schema: dict[str, Any] = {
        "title": "MyOut",
        "type": "object",
        "properties": {
            "a": {"type": "integer", "minimum": 0},
            "b": {
                "anyOf": [
                    {"type": "object", "properties": {"x": {"type": "string"}}, "additionalProperties": True},
                    {"$ref": "#/$defs/RefThing", "description": "should be stripped"},
                ]
            },
            "c": {"type": "array", "items": {"type": "object", "properties": {"y": {"type": "string"}}}},
        },
        "required": ["a"],
        "$defs": {
            "RefThing": {
                "type": "object",
                "properties": {"z": {"type": "string", "maximum": 10}},
            }
        },
    }

    cleaned = OpenAIFormatter._clean_schema_for_openai_strict_mode(schema)

    assert cleaned["type"] == "object"
    assert cleaned["additionalProperties"] is False
    assert set(cleaned["required"]) == {"a", "b", "c"}
    assert "minimum" not in cleaned["properties"]["a"]

    # anyOf branch object is strict
    any_of0 = cleaned["properties"]["b"]["anyOf"][0]
    assert any_of0["additionalProperties"] is False
    assert set(any_of0["required"]) == {"x"}

    # $ref branch keeps only $ref
    any_of1 = cleaned["properties"]["b"]["anyOf"][1]
    assert list(any_of1.keys()) == ["$ref"]

    # Array items object is strict
    assert cleaned["properties"]["c"]["items"]["additionalProperties"] is False


@pytest.mark.case_id("DAI-057")
def test_dai_057_openai_formatter_function_parameters_build_required_and_drop_none():
    """GIVEN FunctionParameters with a mix of required and optional params
    WHEN converted for OpenAI
    THEN additionalProperties is false and required list is auto-derived.
    """

    params = FunctionParameters(
        type="object",
        properties={
            "path": FunctionParameter(type="string", description=None, required=True),
            "limit": FunctionParameter(type="integer", description="max", required=False, default=10),
        },
        required=None,
    )

    converted = OpenAIFormatter.convert_function_parameters(params)
    assert converted["additionalProperties"] is False
    assert converted["required"] == ["path"]
    assert "description" not in converted["properties"]["path"]


@pytest.mark.case_id("DAI-058")
def test_dai_058_openai_message_converter_reasoning_summary_list_and_content_parts():
    """GIVEN an OpenAI Responses-style reasoning output item
    WHEN converted to Dhenara content items
    THEN summary and content parts are preserved for round-tripping.
    """

    class _Summary(SimpleNamespace):
        def model_dump(self):
            return {"type": "summary_text", "text": "s"}

    reasoning = {
        "type": "reasoning",
        "id": "rid",
        "encrypted_content": "sig",
        "status": "complete",
        "summary": [_Summary()],
        "content": [{"type": "thinking", "text": "t1"}, {"type": "thinking", "text": "t2"}],
    }

    ci = OpenAIMessageConverter.provider_message_item_to_dai_content_item(
        message_item=reasoning,
        role="assistant",
        index=0,
        ai_model_provider=AIModelProviderEnum.OPEN_AI,
        structured_output_config=None,
    )

    # Reasoning item text aggregation is implementation-defined; ensure both parts exist.
    txt = ci.get_text()
    assert "t1" in txt
    assert "t2" in txt
    assert ci.thinking_id == "rid"
    assert ci.thinking_signature == "sig"
    assert ci.thinking_status == "complete"
    assert ci.thinking_summary and ci.thinking_summary[0].text == "s"


@pytest.mark.case_id("DAI-059")
def test_dai_059_openai_message_converter_message_structured_output_emits_only_structured_item():
    """GIVEN an OpenAI Responses-style message output containing JSON text
    WHEN structured_output_config is provided
    THEN the converter returns a StructuredOutputContentItem instead of a plain text item.
    """

    class _OutModel(BaseModel):
        a: int

    cfg = StructuredOutputConfig.from_model(_OutModel)

    msg = {
        "type": "message",
        "id": "mid",
        "content": [
            {"type": "output_text", "text": '{"a": 7}', "annotations": []},
        ],
    }

    item = OpenAIMessageConverter.provider_message_item_to_dai_content_item(
        message_item=msg,
        role="assistant",
        index=0,
        ai_model_provider=AIModelProviderEnum.OPEN_AI,
        structured_output_config=cfg,
    )

    # Structured output item has a .structured_output attribute and the original parts
    assert getattr(item, "structured_output", None) is not None
    assert item.structured_output.structured_data == {"a": 7}


@pytest.mark.case_id("DAI-060")
def test_dai_060_openai_message_converter_tool_call_invalid_json_arguments_preserved_as_raw():
    """GIVEN a function_call output item with invalid JSON arguments
    WHEN converted to a tool call content item
    THEN the arguments are preserved under a 'raw' key.
    """

    fc = {
        "type": "function_call",
        "call_id": "c1",
        "id": "id1",
        "name": "file_read",
        "arguments": "{not json}",
    }

    item = OpenAIMessageConverter.provider_message_item_to_dai_content_item(
        message_item=fc,
        role="assistant",
        index=0,
        ai_model_provider=AIModelProviderEnum.OPEN_AI,
        structured_output_config=None,
    )

    assert item.tool_call.name == "file_read"
    assert item.tool_call.arguments["raw"] == "{not json}"
