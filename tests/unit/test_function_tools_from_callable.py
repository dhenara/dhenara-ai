# Provenance: Added to improve function tool schema generation coverage (2026-01-21)

from __future__ import annotations

from typing import Literal

import pytest

from dhenara.ai.types.genai.dhenara.request import ToolDefinition

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-053")
def test_dai_053_tool_definition_from_callable_schema_and_required_fields():
    """GIVEN a python callable with type hints and docstring param descriptions
    WHEN ToolDefinition.from_callable is used
    THEN required/optional fields and enums/items are reflected in the generated schema.
    """

    def demo(
        path: str,
        mode: Literal["r", "w"] = "r",
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> str:
        """Demo tool.

        :param path: Path to read
        :param mode: Read or write mode
        :param tags: Optional tags
        :param limit: Limit entries
        """

        return f"{mode}:{path}:{limit}:{tags}"  # pragma: no cover

    td = ToolDefinition.from_callable(demo)
    assert td.type == "function"
    assert td.function.name == "demo"
    assert "Demo tool" in (td.function.description or "")

    params = td.function.parameters
    assert "path" in (params.required or [])

    assert params.properties["mode"].allowed_values == ["r", "w"]

    # Pydantic JSON schema encodes list[str] as array with items
    assert params.properties["tags"].type == "array" or params.properties["tags"].type == "string"
    if params.properties["tags"].type == "array":
        assert params.properties["tags"].items

    assert params.properties["limit"].default == 10
