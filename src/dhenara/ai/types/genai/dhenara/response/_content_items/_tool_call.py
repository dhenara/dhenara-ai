import json
from typing import Any

from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel


class ChatResponseToolCall(BaseModel):
    """Representation of a tool call from an LLM"""

    call_id: str | None = Field(
        None,
        description="An identifier used to map this tool call to a tool call output.",
    )
    id: str | None = Field(
        None,
        description="The unique ID of the tool call output in the provider platform.",
    )
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_data: str | dict | None = Field(
        None,
        description="Raw unparsed response from the model",
    )
    parse_error: str | None = Field(
        None,
        description="Error that occurred during parsing, if any",
    )

    @classmethod
    def parse_args_str_or_dict(cls, arguments: str | dict) -> dict:
        """Parse arguments from either JSON striing or dict"""
        arguments_dict = {}
        raw_data = None
        parse_error = None
        if isinstance(arguments, str):
            try:
                arguments_dict = json.loads(arguments)
            except Exception as e:
                raw_data = arguments
                parse_error = str(e)
        elif isinstance(arguments, dict):
            try:
                arguments_dict = arguments
            except Exception as e:
                raw_data = arguments
                parse_error = str(e)
        else:
            raw_data = arguments
            parse_error = f"Invalid arguments type {(type(arguments))}"

        return {
            "arguments_dict": arguments_dict,
            "raw_data": raw_data,
            "parse_error": parse_error,
        }


class ChatResponseToolCallResult(BaseModel):
    """Result of executing a tool call, which may pass to LLM in next turn"""

    tool_name: str
    call_id: str | None = None
    result: Any = None
    error: str | None = None
