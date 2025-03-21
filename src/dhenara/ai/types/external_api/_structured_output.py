from typing import Any

from pydantic import BaseModel, Field


class StructuredOutputSchema(BaseModel):
    """Base class for structured output schema"""

    description: str | None = Field(
        default=None,
        description="Description of the output format",
    )

    @classmethod
    def to_json_schema(cls) -> dict[str, Any]:
        """Convert Pydantic model to JSON Schema"""
        _schema = cls.model_json_schema()
        # Clean up some pydantic-specific fields
        if "title" in _schema:
            del _schema["title"]
        return _schema


class StructuredOutputConfig(BaseModel):
    """Configuration for structured output"""

    output_schema: type[StructuredOutputSchema] | dict[str, Any] = Field(
        ...,
        description="Schema for the structured output",
    )
    name: str | None = Field(
        None,
        description="Name for the response format",
    )
    description: str | None = Field(
        None,
        description="Description of what the response format represents",
    )

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI response_format parameter"""
        if isinstance(self.output_schema, type) and issubclass(self.output_schema, StructuredOutputSchema):
            output_schema = self.output_schema.to_json_schema()
        else:
            output_schema = self.output_schema

        return {"type": "json_object", "schema": output_schema}

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool for structured output"""
        if isinstance(self.output_schema, type) and issubclass(self.output_schema, StructuredOutputSchema):
            output_schema = self.output_schema.to_json_schema()
        else:
            output_schema = self.output_schema

        name = self.name or "structured_output"

        return {
            "name": name,
            "description": self.description or "Generate a structured output",
            "input_schema": {"type": "object", "properties": {}, "required": []},
            "output_schema": output_schema,
        }

    def to_google_format(self) -> dict[str, Any]:  # TODO
        """Convert to Google Gemini format"""
        # Google doesn't have a native structured output format,
        # but we can use system instructions
        if isinstance(self.output_schema, type) and issubclass(self.output_schema, StructuredOutputSchema):
            output_schema = self.output_schema.to_json_schema()
        else:
            output_schema = self.output_schema

        return {
            "schema": output_schema,
            # Google may need different handling at the prompt level
        }
