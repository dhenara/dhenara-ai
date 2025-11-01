from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, field_validator

from dhenara.ai.types.shared.base import BaseModel


class ToolCallResult(BaseModel):
    """Represents the output produced by executing a tool call that should be
    supplied back to the model following the provider-specific tool result
    message conventions.
    """

    type: Literal["tool_result"] = "tool_result"
    call_id: str | None = Field(
        default=None,
        description=(
            "Identifier for the originating tool call provided by the model.\n"
            "Optional because some providers (e.g. Google Gemini) don't require it,\n"
        ),
    )
    output: Any = Field(
        default=None,
        description="Structured output from the tool. May be any JSON-serialisable object or string.",
    )
    name: str | None = Field(
        default=None,
        description="Optional tool name hint (used by providers like Google Gemini).",
    )

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

        if isinstance(self.output, (dict, list)):
            return self.output
        if isinstance(self.output, str):
            return {"result": self.output}
        return {"result": self.output}


class ToolCallResultsMessage(BaseModel):
    """Container that groups multiple tool results into a single conversation message."""

    type: Literal["tool_result_message"] = "tool_result_message"
    results: list[ToolCallResult] = Field(
        default_factory=list,
        description="Ordered list of tool call results that originated from the same assistant turn.",
    )

    @field_validator("results")
    @classmethod
    def _validate_results(cls, value: list[ToolCallResult]) -> list[ToolCallResult]:
        if not value:
            raise ValueError("ToolCallResultsMessage.results must contain at least one ToolCallResult")
        return value

    def as_list(self) -> list[ToolCallResult]:
        """Return the underlying list of tool call results."""

        return list(self.results)
