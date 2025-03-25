from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel


class StructuredOutputConfig(BaseModel):
    """Configuration for structured output"""

    # This field will be used for serialization
    output_schema: dict[str, Any] = Field(
        ...,
        description="Schema for the structured output as a JSON schema dictionary",
    )

    # This field will need to be included in the model but marked as excluded from serialization
    model_class_reference: type[PydanticBaseModel] | None = Field(
        None,
        description="Reference to the Pydantic model class (not serialized)",
        exclude=True,  # This ensures it's excluded from serialization
    )

    @classmethod
    def from_model(cls, model_class: type[PydanticBaseModel]):
        """Create config from a model class"""
        return cls(
            output_schema=model_class.model_json_schema(),
            model_class_reference=model_class,  # Store the reference for programmatic access
        )

    def get_schema(self) -> dict[str, Any]:
        """Get the schema object"""
        return self.output_schema

    def get_model_class(self) -> type[PydanticBaseModel] | None:
        """Get the model class for Python operations"""
        return self.model_class_reference

    # Override model_dump to handle the model_class field
    def model_dump(self, **kwargs):
        # Get the default serialization
        data = super().model_dump(**kwargs)
        # Remove model_class if it's in the output
        if "model_class_reference" in data:
            del data["model_class_reference"]
        return data
