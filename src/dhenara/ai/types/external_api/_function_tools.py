from typing import Any, Literal

from pydantic import BaseModel, Field

from dhenara.ai.types.external_api import AIModelProviderEnum


class ProviderConversionMixin:
    def to_provider_format(self, provider: AIModelProviderEnum) -> dict[str, Any]:
        """Convert to provider format"""
        if provider == AIModelProviderEnum.OPEN_AI:
            return self.to_openai_format()
        elif provider == AIModelProviderEnum.ANTHROPIC:
            return self.to_anthropic_format()
        elif provider == AIModelProviderEnum.GOOGLE_AI:
            return self.to_google_format()
        else:
            raise ValueError(f"Provider {provider} not supported")


class FunctionParameter(BaseModel, ProviderConversionMixin):
    """Parameter definition for function/tool parameters"""

    type: str = Field(..., description="Type of the parameter (string, number, boolean, etc.)")
    description: str | None = Field(default=None, description="Description of the parameter")
    required: bool = Field(default=False, description="Whether the parameter is required")
    allowed_values: list[Any] | None = Field(default=None, description="Allowed values")
    default: Any | None = Field(default=None, description="Default value for the parameter")

    def to_openai_format(self) -> dict[str, Any]:
        result = self.model_dump(
            exclude={"required", "allowed_values", "default"},
        )
        return result

    def to_anthropic_format(self) -> dict[str, Any]:
        result = self.model_dump(
            exclude={"required", "allowed_values", "default"},
        )
        if self.allowed_values is not None:
            result["enum"] = self.allowed_values

        return result

    def to_google_format(self) -> dict[str, Any]:
        result = self.model_dump(
            exclude={"required", "allowed_values", "default"},
        )
        return result


class FunctionParameters(BaseModel, ProviderConversionMixin):
    """Schema for function parameters"""

    type: Literal["object"] = "object"
    properties: dict[str, FunctionParameter] = Field(..., description="Properties of the function parameters")
    required: list[str] | None = Field(default_factory=list, description="List of required parameters")

    # TODO: Add a `from_callable` fn

    def _to_common_format(self, provider: AIModelProviderEnum) -> dict[str, Any]:
        """Convert to OpenAI format"""
        # Create a new dictionary with transformed properties
        result = {
            "type": self.type,
            "properties": {name: param.to_provider_format(provider) for name, param in self.properties.items()},
        }

        # Auto-build the required list based on parameters marked as required
        required_params = [name for name, param in self.properties.items() if param.required]

        # Only include required field if there are required parameters
        if required_params:
            result["required"] = required_params
        elif self.required:  # If manually specified required array exists
            result["required"] = self.required

        return result

    def to_openai_format(self) -> dict[str, Any]:
        return self._to_common_format(AIModelProviderEnum.OPEN_AI)

    def to_anthropic_format(self) -> dict[str, Any]:
        return self._to_common_format(AIModelProviderEnum.ANTHROPIC)

    def to_google_format(self) -> dict[str, Any]:
        return self._to_common_format(AIModelProviderEnum.GOOGLE_AI)


class FunctionDefinition(BaseModel, ProviderConversionMixin):
    """Generic function/tool definition that works across all providers"""

    name: str = Field(..., description="Name of the function")
    description: str | None = Field(default=None, description="Description of the function")
    parameters: FunctionParameters = Field(..., description="Parameters for the function")

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_openai_format(),
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic format"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters.to_anthropic_format(),
        }

    def to_google_format(self) -> dict[str, Any]:
        """Convert to Google Gemini format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_google_format(),
        }


class ToolDefinition(BaseModel, ProviderConversionMixin):
    """Tool definition that wraps a function"""

    type: Literal["function"] = "function"
    function: FunctionDefinition = Field(..., description="Function definition")

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": self.function.to_openai_format(),
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        return self.function.to_anthropic_format()

    def to_google_format(self) -> dict[str, Any]:
        return {"function_declarations": [self.function.to_google_format()]}


# TODO: Add privider specific fns
class BuiltInTool(BaseModel):
    """Built-in Tool by provider"""

    type: str
    name: str


class ToolChoice(BaseModel):
    type: Literal["zero_or_more", "one_or_more", "specific"] | None = Field(
        default="zero_or_more",
        description=(
            "Tool choice type. "
            "NOTE: A `None` will make disable all tolls and responses will be as if like without a toll"
        ),
    )
    specific_tool_name: str | None = None

    def to_openai_format(self) -> dict[str, Any]:
        if self.type is None:
            return None
        elif self.type == "zero_or_more":  # Auto: (Default) Call zero, one, or multiple functions. tool_choice: "auto"
            return "auto"
        elif self.type == "one_or_more":  # Required: Call one or more functions. tool_choice: "required"
            return "required"
        elif self.type == "specific":  # Forced Function: Call exactly one specific function. tool_choice:
            return {"type": "function", "name": self.specific_tool_name}

    def to_anthropic_format(self) -> dict[str, Any]:
        if self.type is None:
            return None
        elif self.type == "zero_or_more":
            return {"type": "auto"}
        elif self.type == "one_or_more":
            return {"type": "any"}
        elif self.type == "specific":
            return {"type": "tool", "name": self.specific_tool_name}

    def to_google_format(self) -> dict[str, Any]:
        if self.type is None:
            return None
        elif self.type == "zero_or_more":
            _cfg = {
                "mode": "AUTO",
            }
        elif self.type == "one_or_more":
            _cfg = {
                "mode": "ANY",
            }
        elif self.type == "specific":
            _cfg = {
                "mode": "AUTO",
                "allowed_function_names": [self.specific_tool_name],
            }

        return {"function_calling_config": _cfg}
