from typing import Any

from pydantic import Field, model_validator

from dhenara.ai.types.shared.base import BaseModel


class TextTemplateVariableProps(BaseModel):
    default_value: Any | None = Field(default=None, description="Default value for the parameter")
    allowed_values: list[Any] | None = Field(default=None, description="Allowed values")
    # type: str = Field(..., description="Type of the parameter (string, number, boolean, etc.)")
    # description: str | None = Field(default=None, description="Description of the parameter")
    # required: bool = Field(default=False, description="Whether the parameter is required")


class TextTemplate(BaseModel):
    """Template configuration for AI interactions.

    A generic template structure for configuring various AI interaction texts,
    including prompts, system instructions, context information, or any templated text.
    Supports dynamic variable substitution through standard Python string formatting.
    """

    text: str = Field(
        description="Text template with optional {placeholders} for string formatting",
    )
    variables: dict[str, TextTemplateVariableProps | None] = Field(
        default_factory=dict,
        description="Variables/parameters for the template",
    )

    @model_validator(mode="after")
    def validate_variables(self) -> "TextTemplate":
        """Validate that all variables are present in the template text and vice versa."""

        text = self.text
        template_args = list(self.variables.keys())

        # Check for variables that don't appear in the text
        # Use string.Formatter to extract field names from the template
        import string

        formatter = string.Formatter()
        field_names = [field_name for _, field_name, _, _ in formatter.parse(text) if field_name is not None]

        # Check for variables defined but not used in template
        missing_in_text = [arg for arg in template_args if arg not in field_names]
        if missing_in_text:
            raise ValueError(f"Variables {missing_in_text} are defined but not used in the template text")

        # Check for placeholders in the text that aren't defined in variables
        extra_in_text = [field for field in field_names if field not in template_args]
        if extra_in_text:
            raise ValueError(f"Placeholders {extra_in_text} are used in the template text but not defined in variables")

        return self

    def get_args_default_values(self) -> dict[str, Any]:
        """Get a dictionary of variable default values."""
        return {key: props.default_value for key, props in self.variables.items() if props.default_value is not None}

    def format(self, **kwargs) -> str:
        """Formats the template with provided values.

        Additional keyword variables can be passed at runtime to be included
        in the template formatting, overriding any matching keys in the
        default_values dictionary.

        Args:
            **kwargs: Runtime values for template variables (overrides defaults)

        Returns:
            str: The complete formatted text
        """
        # Start with default values and override with runtime values
        format_values = self.get_args_default_values()
        format_values.update(kwargs)

        # Check if all required variables are provided
        if self.variables:
            missing_args = [arg for arg in self.variables.keys() if arg not in format_values]
            if missing_args:
                raise ValueError(f"Missing required variables: {missing_args}")

        # Handle special placeholders
        # TODO_FUTURE: Use `jinja2` for processing specail jinja2 and remove if not present
        # special_placeholders = {
        #    "dh_input_content": self._handle_input_content,
        #    # more special placeholders and their handlers here
        # }

        # Format template with variables
        return self.text.format(**format_values)
