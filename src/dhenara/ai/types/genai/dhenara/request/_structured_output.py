import logging
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel

logger = logging.getLogger(__name__)


class StructuredOutputConfig(BaseModel):
    """Configuration for structured output"""

    output_schema: type[PydanticBaseModel] | dict[str, Any] = Field(
        ...,
        description="Schema for the structured output",
    )

    def _get_schema(self) -> dict[str, Any]:
        schema = None
        if isinstance(self.output_schema, type) and issubclass(self.output_schema, PydanticBaseModel):
            schema = self.output_schema.model_json_schema()
        elif isinstance(self.output_schema, PydanticBaseModel):
            schema = self.output_schema.model_json_schema()
        elif isinstance(self.output_schema, dict):
            schema = self.output_schema
        else:
            raise ValueError(f"Unknown output_schema type {type(self.output_schema)} ")

        return schema

    def to_openai_format(self) -> dict[str, Any]:
        # Get the original JSON schema from Pydantic or dict
        schema = self._get_schema()

        # Extract the name from the title and remove the title key.
        if "title" in schema:
            schema_name = schema.pop("title")  # :NOTE: pop
        else:
            schema_name = "output"

        # Remove JSON Schema keywords that OpenAI doesn't permit
        if "properties" in schema:
            for prop in schema["properties"].values():
                # Remove numeric constraints that are not permitted.from pydantic ge, le
                prop.pop("minimum", None)
                prop.pop("maximum", None)

        # Ensure the schema has additionalProperties set to False
        schema.setdefault("additionalProperties", False)

        # Add the name key required by OpenAI
        schema["name"] = schema_name

        # Optionally, if you want to enforce strict mode on the overall schema,
        # you can wrap it or add a flag. In this example, we simply return the modified schema.
        result = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        }
        return result

    def to_google_format(self) -> dict[str, Any]:  # TODO
        return self.output_schema

    def to_anthropic_format(self) -> dict[str, Any]:
        """
        Convert structured output config to Anthropic tool format.
        Since Anthropic doesn't directly support structured output,
        we create a specialized tool and force the model to use it.
        """
        schema = self._get_schema()
        name = schema.pop("title", None) or "structured_output"
        description = "Generate structured output according to schema"

        # Clean up schema for Anthropic compatibility
        if "properties" in schema:
            for prop in schema["properties"].values():
                # Remove unsupported validations
                for field in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
                    if field in prop:
                        prop.pop(field)

                # Convert enum to Anthropic format
                if "enum" in prop:
                    prop["enum"] = list(prop["enum"])

        # Create the tool
        tool = {
            "name": name,
            "description": description,
            "input_schema": schema,
        }

        return tool
