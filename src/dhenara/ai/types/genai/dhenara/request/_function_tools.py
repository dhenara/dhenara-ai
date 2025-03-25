import inspect
from collections.abc import Callable
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, Field, create_model

from dhenara.ai.types.genai.dhenara.request import ToProviderMixin


class FunctionParameter(BaseModel, ToProviderMixin):
    """Parameter definition for function/tool parameters"""

    type: str = Field(..., description="Type of the parameter (string, number, boolean, etc.)")
    description: str | None = Field(default=None, description="Description of the parameter")
    required: bool = Field(default=False, description="Whether the parameter is required")
    allowed_values: list[Any] | None = Field(default=None, description="Allowed values")
    default: Any | None = Field(default=None, description="Default value for the parameter")


class FunctionParameters(BaseModel, ToProviderMixin):
    """Schema for function parameters"""

    type: Literal["object"] = "object"
    properties: dict[str, FunctionParameter] = Field(..., description="Properties of the function parameters")
    required: list[str] | None = Field(default_factory=list, description="List of required parameters")


class FunctionDefinition(BaseModel, ToProviderMixin):
    """Generic function/tool definition that works across all providers"""

    name: str = Field(..., description="Name of the function")
    description: str | None = Field(default=None, description="Description of the function")
    parameters: FunctionParameters = Field(..., description="Parameters for the function")


class ToolDefinition(BaseModel, ToProviderMixin):
    """Tool definition that wraps a function"""

    type: Literal["function"] = "function"
    function: FunctionDefinition = Field(..., description="Function definition")
    function_reference: Callable | None = Field(default=None, description="Callable fn reference")

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
        return cls(function=function_def, function_reference=func)


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
