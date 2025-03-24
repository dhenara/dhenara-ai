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
