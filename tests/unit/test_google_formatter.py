# Provenance: Added to prevent Google/Gemini tool-schema regressions (2026-04-19)

from __future__ import annotations

from typing import Any, cast

import pytest
from google.genai.types import Tool

from dhenara.ai.providers.google.formatter import GoogleFormatter
from dhenara.ai.types.genai.dhenara.request import FunctionDefinition, FunctionParameter, FunctionParameters

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-300")
def test_dai_300_google_formatter_rewrites_oneof_array_items_for_tools():
    """GIVEN a tool schema with oneOf under array items
    WHEN GoogleFormatter converts the function definition
    THEN oneOf is rewritten to anyOf with full typed branches so Gemini accepts the tool declaration.
    """

    params = FunctionParameters(
        type="object",
        properties={
            "ranges": FunctionParameter(
                type="array",
                description="Line ranges to read.",
                required=False,
                items={
                    "type": "object",
                    "properties": {
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "length": {"type": "integer"},
                    },
                    "required": ["start_line"],
                    "oneOf": [
                        {"required": ["end_line"]},
                        {"required": ["length"]},
                    ],
                    "additionalProperties": False,
                },
            )
        },
    )
    converted = GoogleFormatter.convert_function_definition(
        FunctionDefinition(
            name="file_read",
            description="Read one or more file ranges.",
            parameters=params,
        )
    )

    items = converted["parameters"]["properties"]["ranges"]["items"]
    assert "oneOf" not in items
    assert "anyOf" in items
    assert "type" not in items

    first_branch = items["anyOf"][0]
    second_branch = items["anyOf"][1]

    assert first_branch["type"] == "object"
    assert second_branch["type"] == "object"
    assert set(first_branch["properties"]) == {"start_line", "end_line", "length"}
    assert set(second_branch["properties"]) == {"start_line", "end_line", "length"}
    assert first_branch["required"] == ["start_line", "end_line"]
    assert second_branch["required"] == ["start_line", "length"]
    assert "additionalProperties" not in first_branch
    assert "additionalProperties" not in second_branch

    tool = Tool(function_declarations=cast(Any, [converted]))
    dumped = tool.model_dump(exclude_none=True, by_alias=True)
    dumped_items = dumped["functionDeclarations"][0]["parameters"]["properties"]["ranges"]["items"]["anyOf"]
    assert dumped_items[0]["type"] == "OBJECT"
    assert dumped_items[1]["type"] == "OBJECT"
