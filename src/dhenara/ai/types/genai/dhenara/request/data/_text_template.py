from typing import Any

from pydantic import Field, model_validator

from dhenara.ai.types.shared.base import BaseModel


class TextTemplateVariableProps(BaseModel):
    default: Any | None = Field(default=None, description="Default value for the parameter")
    allowed: list[Any] | None = Field(default=None, description="Allowed values")
    # type: str = Field(..., description="Type of the parameter (string, number, boolean, etc.)")
    # description: str | None = Field(default=None, description="Description of the parameter")
    # required: bool = Field(default=False, description="Whether the parameter is required")


class TextTemplate(BaseModel):
    """
    Enhanced template configuration for AI interactions.
    Supports two template syntaxes:

    1. Variable substitution with $var{variable_name}
    2. Expression-based templates with $expr{expressions}

    Regular braces {} are left untouched and treated as literal text.

    Escape sequences:
    - Use $$var{} to output a literal "$var{}" string
    - Use $$expr{} to output a literal "$expr{}" string

    NOTE: Parsing templates using this syntaxes are NOT handled within this (dhenara.ai) package.
    You need to use a separate template engine for that.
    """

    text: str = Field(
        description="Text template with $var{variables} and $expr{expressions}",
    )
    variables: dict[str, TextTemplateVariableProps | None] = Field(
        default_factory=dict,
        description="Variables/parameters for the template",
    )
    disable_checks: bool = Field(
        default=False,
    )

    @model_validator(mode="after")
    def validate_variables(self) -> "TextTemplate":
        """
        Validate that all defined variables appear in the template.
        Looks for $var{variable_name} patterns in the template text.
        Supports variables with dot notation (nested variables).
        """
        if not self.disable_checks:
            import re

            # Get variable names from the template using $var{} pattern
            text = self.text
            defined_vars = list(self.variables.keys())

            # Find all $var{...} patterns, but ignore escaped ones ($$var{...})
            # First replace all escaped patterns with a placeholder
            escaped_patterns = {}

            def replace_escaped(match):
                placeholder = f"__ESCAPED_{len(escaped_patterns)}__"
                escaped_patterns[placeholder] = match.group(0)
                return placeholder

            # Replace escaped patterns
            text_for_validation = re.sub(r"\$\$var{[^}]*}", replace_escaped, text)
            text_for_validation = re.sub(r"\$\$expr{[^}]*}", replace_escaped, text_for_validation)

            # Now find all non-escaped variable patterns
            var_pattern = re.compile(r"\$var{([^}]+)}")
            used_vars_full = [match.group(1).strip() for match in var_pattern.finditer(text_for_validation)]

            # Extract root variable names from possibly nested paths (e.g., 'task_spec.description' -> 'task_spec')
            used_vars = set()
            for var_path in used_vars_full:
                root_var = var_path.split(".")[0].strip()
                used_vars.add(root_var)

            # Check for variables defined but not used in template
            missing_in_text = [var for var in defined_vars if var not in used_vars]
            if missing_in_text:
                raise ValueError(f"Variables {missing_in_text} are defined but not used in the template text")

        return self

    def get_variable_names(self) -> dict[str, Any]:
        """Get a dictionary of variable default values."""
        return list(self.variables.keys())

    def get_args_default_values(self) -> dict[str, Any]:
        """Get a dictionary of variable default values."""
        return {key: props.default for key, props in self.variables.items() if props and props.default is not None}

    def format(self, **kwargs) -> str:
        """
        Returns the template text without processing any variables or expressions.
        Variable substitution and expression evaluation are NOT handled within this package.

        This is NOT a formatting method - it just returns the raw template.
        """
        # Just return the raw template text - no Python formatting
        return self.text


class ObjectTemplate(BaseModel):
    """
    Template configuration for retrieving objects from expressions.
    Unlike TextTemplate, this preserves the type of the evaluated expression
    rather than converting to string.

    Uses the $expr{} syntax for expressions.
    """

    expression: str = Field(
        description="Expression template containing a $expr{expression} that returns an object",
    )
