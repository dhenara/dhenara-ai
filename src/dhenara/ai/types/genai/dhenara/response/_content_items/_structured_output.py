import logging
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from dhenara.ai.types.genai.dhenara import StructuredOutputConfig
from dhenara.ai.types.shared.base import BaseModel

from ._tool_call import ChatResponseToolCall

logger = logging.getLogger(__name__)


class ChatResponseStructuredOutput(BaseModel):
    """Content item specific to structured output responses

    Contains the structured data output from the model according to a specified schema

    Attributes:
        type: The type of content item (always STRUCTURED_OUTPUT)
        structured_data: The parsed structured data
        raw_data: The raw unparsed response from the model
        schema: The schema that was used for the structured output
        parse_error: Any error that occurred during parsing
    """

    config: StructuredOutputConfig = Field(
        ...,
        description="StructuredOutputConfig used for generating this response",
    )
    structured_data: Any = Field(
        None,
        description="Parsed structured data according to the schema",
    )
    raw_data: str | dict | None = Field(
        None,
        description="Raw unparsed response from the model",
    )
    parse_error: str | None = Field(
        None,
        description="Error that occurred during parsing, if any",
    )

    def get_text(self) -> str:
        """Get a text representation of the structured data"""
        if self.structured_data is not None:
            return str(self.structured_data)
        elif self.raw_data is not None:
            return str(self.raw_data)
        elif self.parse_error is not None:
            return f"Error parsing structured output: {self.parse_error}"
        return ""

    def as_pydantic(self, model_class: type[PydanticBaseModel] | None = None) -> PydanticBaseModel | None:
        """Convert the structured data to a pydantic model instance

        Args:
            model_class: Optional pydantic model class to use for conversion.
                         If not provided, uses the original schema class if available.

        Returns:
            Pydantic model instance or None if conversion fails
        """
        if self.structured_data is None:
            return None

        try:
            if model_class is not None:
                return model_class.model_validate(self.structured_data)
            return self.structured_data  # If already a pydantic model
        except Exception as e:
            logger.error(f"Error converting structured data to pydantic model: {e}")
            return None

    @classmethod
    def from_model_output(
        cls,
        raw_response: str | dict,
        config: StructuredOutputConfig,
    ) -> "ChatResponseStructuredOutput":
        """Create a structured output content item from model output

        Args:
            output: Raw output from the model
            schema: Schema to validate against
            role: Role of the message sender
            index: Index of the content item

        Returns:
            ChatResponseStructuredOutputContentItem
        """

        error = None
        model_cls: type[PydanticBaseModel] = None
        if isinstance(config.output_schema, type) and issubclass(config.output_schema, PydanticBaseModel):
            model_cls = config.output_schema
        elif isinstance(config.output_schema, PydanticBaseModel):
            model_cls = config.output_schema.__class__
        else:
            error = "from_model_output: Unsupported schema type for structured output."

        try:
            # Attempt to load and validate the response
            parsed_data = model_cls.model_validate_json(raw_response)
        except Exception as e:
            logger.exception(f"parse_response: Error: {e}")
            error = str(e)

        return cls(
            config=config,
            structured_data=parsed_data if error is None else None,
            raw_data=raw_response,
            parse_error=error,
        )

    @classmethod
    def from_tool_call(
        cls,
        tool_call: ChatResponseToolCall,
        config: StructuredOutputConfig,
    ) -> "ChatResponseStructuredOutput":
        """Create a structured output from a tool call response

        Args:
            tool_call: The tool call response
            config: StructuredOutputConfig to use for validation

        Returns:
            ChatResponseStructuredOutput instance
        """
        error = None
        parsed_data = None
        raw_response = tool_call.arguments.arguments_dict  # Get the dict directly

        model_cls: type[PydanticBaseModel] = None
        if isinstance(config.output_schema, type) and issubclass(config.output_schema, PydanticBaseModel):
            model_cls = config.output_schema
        elif isinstance(config.output_schema, PydanticBaseModel):
            model_cls = config.output_schema.__class__
        else:
            error = "from_tool_call: Unsupported schema type for structured output."

        try:
            # Use model_validate instead of model_validate_json since we already have a dict
            if error is None:
                parsed_data = model_cls.model_validate(raw_response)
        except Exception as e:
            logger.exception(f"parse_response: Error: {e}")
            error = str(e)

        return cls(
            config=config,
            structured_data=parsed_data if error is None else None,
            raw_data=raw_response,
            parse_error=error,
        )
