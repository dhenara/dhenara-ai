import json
from typing import Any

from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel


class ChatResponseToolCallArguments(BaseModel):
    """Arguments for a tool call received from an LLM"""

    arguments_dict: dict[str, Any] = Field(default_factory=dict)
    raw_json: str | None = None

    @classmethod
    def parse(cls, arguments: str | dict) -> "ChatResponseToolCallArguments":
        """Parse arguments from either JSON striing or dict"""
        if isinstance(arguments, str):
            try:
                return cls(arguments_dict=json.loads(arguments), raw_json=arguments)
            except json.JSONDecodeError:
                return cls(arguments_dict={}, raw_json=arguments)
        return cls(arguments_dict=arguments)


class ChatResponseToolCall(BaseModel):
    """Representation of a tool call from an LLM"""

    id: str | None = None
    name: str
    arguments: ChatResponseToolCallArguments

    @classmethod
    def from_openai_format(cls, props: dict) -> "ChatResponseToolCall":
        """Create from OpenAI tool call format"""
        return cls(
            id=props.get("id"),
            name=props.get("function", {}).get("name"),
            arguments=ChatResponseToolCallArguments.parse(props.get("function", {}).get("arguments")),
        )

    @classmethod
    def from_anthropic_format(cls, props: dict) -> "ChatResponseToolCall":
        """Create from Anthropic tool use format"""
        return cls(
            id=props.get("id"),
            name=props.get("name"),
            arguments=ChatResponseToolCallArguments.parse(props.get("input")),
        )

    @classmethod
    def from_google_format(cls, props: dict) -> "ChatResponseToolCall":
        return cls(
            id=props.get("id"),
            name=props.get("name"),
            arguments=ChatResponseToolCallArguments.parse(props.get("args")),
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
