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
    Supports both Python style formatting with {placeholders} and
    expression-based templates with ${expressions}.
    Note that ${expressions} are not parsed within this package
    """

    text: str = Field(
        description="Text template with optional {placeholders} and ${expressions}",
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
        if not self.disable_checks:
            """Validate that all Python-style variables are present in the template text."""
            # Original validation for Python-style formatting
            text = self.text
            python_template_args = list(self.variables.keys())

            # Check for variables that don't appear in the text
            # Use string.Formatter to extract field names from the template
            import string

            formatter = string.Formatter()
            field_names = [field_name for _, field_name, _, _ in formatter.parse(text) if field_name is not None]

            # Check for variables defined but not used in template
            missing_in_text = [arg for arg in python_template_args if arg not in field_names]
            if missing_in_text:
                raise ValueError(f"Variables {missing_in_text} are defined but not used in the template text")

            # Check for placeholders in the text that aren't defined in variables
            extra_in_text = [field for field in field_names if field not in python_template_args]
            if extra_in_text:
                raise ValueError(
                    f"Placeholders {extra_in_text} are used in the template text but not defined in variables"
                )

        return self

    def get_args_default_values(self) -> dict[str, Any]:
        """Get a dictionary of variable default values."""
        return {key: props.default for key, props in self.variables.items() if props and props.default is not None}

    # def get_args_default_values(self) -> dict[str, Any]:
    #    """Get a dictionary of variable default values."""
    #    return {
    #        key: props.get("default")
    #        for key, props in self.variables.items()
    #        if props and "default" in props
    #    }

    def format(self, **kwargs) -> str:
        """
        Formats the template with provided values, supporting both Python-style
        placeholders and expression-based templates.
        Additional keyword variables can be passed at runtime to be included
        in the template formatting, overriding any matching keys in the

        Args:
            **kwargs: Runtime values for template variables (overrides defaults)

        Returns:
            str: The complete formatted text
        """
        # First apply Python-style formatting
        format_values = self.get_args_default_values()
        format_values.update(kwargs)

        # Check if all required variables are provided
        if not self.disable_checks and self.variables:
            missing_args = [arg for arg in self.variables.keys() if arg not in format_values]
            if missing_args:
                raise ValueError(f"Missing required variables: {missing_args}")

        # Apply Python-style formatting
        # TODO_FUTURE: Use `jinja2` for processing specail jinja2 and remove if not present
        # special_placeholders = {
        #    "dh_input_content": self._handle_input_content,
        #    # more special placeholders and their handlers here
        # }
        result = self.text.format(**format_values)

        #  Note: evaluating expressions are not within the scope of this package
        # It should be taken care seperately.
        return result


class ObjectTemplate(BaseModel):
    """
    Template configuration for retrieving objects from expressions.
    Unlike TextTemplate, this preserves the type of the evaluated expression
    rather than converting to string.
    """

    expression: str = Field(
        description="Expression template containing a ${expression} that returns an object",
    )
