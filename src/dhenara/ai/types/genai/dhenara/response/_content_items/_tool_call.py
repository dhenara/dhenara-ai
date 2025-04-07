import json
from typing import Any

from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel


class ChatResponseToolCall(BaseModel):
    """Representation of a tool call from an LLM"""

    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_data: str | dict | None = Field(
        None,
        description="Raw unparsed response from the model",
    )
    parse_error: str | None = Field(
        None,
        description="Error that occurred during parsing, if any",
    )

    @classmethod
    def _parse(cls, arguments: str | dict) -> dict:
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

    @classmethod
    def from_openai_format(cls, props: dict) -> "ChatResponseToolCall":
        """Create from OpenAI tool call format"""
        _args = props.get("function", {}).get("arguments")
        _parse_result = cls._parse(_args)

        return cls(
            id=props.get("id"),
            name=props.get("function", {}).get("name"),
            arguments=_parse_result.get("arguments_dict"),
            raw_data=_parse_result.get("raw_data"),
            parse_error=_parse_result.get("parse_error"),
        )

    @classmethod
    def from_anthropic_format(cls, props: dict) -> "ChatResponseToolCall":
        """Create from Anthropic tool use format"""
        _args = props.get("input")
        _parse_result = cls._parse(_args)

        return cls(
            id=props.get("id"),
            name=props.get("name"),
            arguments=_parse_result.get("arguments_dict"),
            raw_data=_parse_result.get("raw_data"),
            parse_error=_parse_result.get("parse_error"),
        )

    @classmethod
    def from_google_format(cls, props: dict) -> "ChatResponseToolCall":
        _args = props.get("args")
        _parse_result = cls._parse(_args)

        return cls(
            id=props.get("id"),
            name=props.get("name"),
            arguments=_parse_result.get("arguments_dict"),
            raw_data=_parse_result.get("raw_data"),
            parse_error=_parse_result.get("parse_error"),
        )


class ChatResponseToolCallResult(BaseModel):
    """Result of executing a tool call, which may pass to LLM in next turn"""

    tool_name: str
    call_id: str | None = None
    result: Any = None
    error: str | None = None

    def to_openai_format(self) -> dict:
        """Convert to OpenAI format for tool response"""
        return {
            "tool_call_id": self.call_id,
            "role": "tool",
            "name": self.tool_name,
            "content": json.dumps(self.result) if self.result is not None else str(self.error),
        }

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic format for tool response"""
        return {
            "type": "tool_result",
            "tool_use_id": self.call_id,
            "content": json.dumps(self.result) if self.result is not None else str(self.error),
        }
