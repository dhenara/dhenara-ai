from __future__ import annotations

import json
from typing import Any

from pydantic import Field, field_validator

from dhenara.ai.types.shared.base import BaseModel


class ToolCallResult(BaseModel):
    """Represents the output produced by executing a tool call that should be
    supplied back to the model following the provider-specific tool result
    message conventions.
    """

    type: str = Field(
        default="tool_result",
        description="Discriminator to simplify downstream detection of tool call results.",
    )
    call_id: str = Field(
        ...,
        description="Identifier for the originating tool call provided by the model.",
    )
    output: Any = Field(
        default=None,
        description="Structured output from the tool. May be any JSON-serialisable object or string.",
    )
    name: str | None = Field(
        default=None,
        description="Optional tool name hint (used by providers like Google Gemini).",
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value != "tool_result":
            raise ValueError("ToolCallResult.type must be 'tool_result'")
        return value

    def as_text(self) -> str:
        """Render the output as a text snippet suitable for providers that expect string payloads."""

        if isinstance(self.output, str):
            return self.output
        try:
            return json.dumps(self.output, ensure_ascii=False)
        except TypeError:
            return json.dumps(str(self.output), ensure_ascii=False)

    def as_json(self) -> Any:
        """Render the output as a JSON-compatible object suitable for providers expecting dict/list."""

        if isinstance(self.output, (dict, list)):  # type: ignore[arg-type]
            return self.output
        if isinstance(self.output, str):
            return {"result": self.output}
        return {"result": self.output}
