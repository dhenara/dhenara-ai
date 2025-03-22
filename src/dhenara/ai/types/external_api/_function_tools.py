import inspect
from collections.abc import Callable
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, Field, create_model

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

    @classmethod
    def from_callable(cls, func: Callable) -> "ToolDefinition":
        """
        Create a ToolDefinition from a Python callable.

        Args:
            func: A Python function to convert to a tool definition

        Returns:
            A ToolDefinition object representing the function
        """
        # Get function signature and docstring
        signature = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # Get type hints
        type_hints = get_type_hints(func)

        # Create field definitions for Pydantic model
        fields = {}
        required_params = []

        for name, param in signature.parameters.items():
            if name == "self":
                continue

            # Get type annotation
            annotation = type_hints.get(name, Any)

            # Get parameter description
            param_desc = None
            for line in doc.split("\n"):
                if f":param {name}:" in line:
                    param_desc = line.split(f":param {name}:")[1].strip()
                    break

            # Determine if parameter is required
            is_required = param.default == inspect.Parameter.empty
            default = ... if is_required else param.default

            if is_required:
                required_params.append(name)

            # Add field to model
            fields[name] = (annotation, Field(default=default, description=param_desc))

        # Create a Pydantic model dynamically
        _params_model = create_model(f"{func.__name__}Params", **fields)

        # Get JSON schema from Pydantic model
        schema = _params_model.model_json_schema()

        # Convert to FunctionParameters
        properties = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            json_type = prop_schema.get("type", "string")
            description = prop_schema.get("description")

            properties[prop_name] = FunctionParameter(
                type=json_type,
                description=description,
                required=prop_name in required_params,
                default=None if prop_name in required_params else signature.parameters[prop_name].default,
                allowed_values=prop_schema.get("enum"),
            )

        # Create function definition
        function_def = FunctionDefinition(
            name=func.__name__,
            description=doc.split("\n\n")[0] if doc else None,  # First paragraph of docstring
            parameters=FunctionParameters(properties=properties, required=required_params),
        )

        # Create and return tool definition
        return cls(function=function_def)


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
