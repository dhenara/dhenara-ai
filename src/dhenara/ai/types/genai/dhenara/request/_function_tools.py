import inspect
from collections.abc import Callable
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, Field, create_model


class FunctionParameter(BaseModel):
    """Parameter definition for function/tool parameters"""

    type: str = Field(..., description="Type of the parameter (string, number, boolean, etc.)")
    format: str | None = Field(default=None, description="Optional JSON schema format such as date or date-time")
    description: str | None = Field(default=None, description="Description of the parameter")
    required: bool = Field(default=False, description="Whether the parameter is required")
    allowed_values: list[Any] | None = Field(default=None, description="Allowed values")
    default: Any | None = Field(default=None, description="Default value for the parameter")
    items: dict | None = Field(
        default=None,
        description="JSON schema for array items when type == 'array' (e.g., {'type':'string'})",
    )


def _collapse_optional_union_schema(prop_schema: dict[str, Any]) -> dict[str, Any]:
    for union_key in ("anyOf", "oneOf"):
        raw_variants = prop_schema.get(union_key)
        if not isinstance(raw_variants, list):
            continue
        variants = [variant for variant in raw_variants if isinstance(variant, dict)]
        non_null_variants = [variant for variant in variants if variant.get("type") != "null"]
        if len(non_null_variants) != 1:
            continue
        merged = dict(non_null_variants[0])
        for key in ("description", "default", "title", "examples"):
            if key in prop_schema and key not in merged:
                merged[key] = prop_schema[key]
        return merged
    return prop_schema


def function_parameter_from_json_schema(
    prop_schema: dict[str, Any],
    *,
    required: bool,
    default: Any | None,
    description: str | None = None,
) -> FunctionParameter:
    normalized_schema = _collapse_optional_union_schema(prop_schema)
    return FunctionParameter(
        type=str(normalized_schema.get("type") or prop_schema.get("type") or "string"),
        format=normalized_schema.get("format") or prop_schema.get("format"),
        description=description if description is not None else normalized_schema.get("description"),
        required=required,
        default=default,
        allowed_values=normalized_schema.get("enum") or prop_schema.get("enum"),
        items=normalized_schema.get("items") or prop_schema.get("items"),
    )


class FunctionParameters(BaseModel):
    """Schema for function parameters"""

    type: Literal["object"] = "object"
    properties: dict[str, FunctionParameter] = Field(..., description="Properties of the function parameters")
    required: list[str] | None = Field(default_factory=list, description="List of required parameters")


class FunctionDefinition(BaseModel):
    """Generic function/tool definition that works across all providers"""

    name: str = Field(..., description="Name of the function")
    description: str | None = Field(default=None, description="Description of the function")
    parameters: FunctionParameters = Field(..., description="Parameters for the function")


class ToolDefinition(BaseModel):
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
            properties[prop_name] = function_parameter_from_json_schema(
                prop_schema,
                required=prop_name in required_params,
                default=None if prop_name in required_params else signature.parameters[prop_name].default,
                description=prop_schema.get("description"),
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
