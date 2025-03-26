import json
import logging
from typing import TypeVar

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from dhenara.ai.types.genai.dhenara import StructuredOutputConfig
from dhenara.ai.types.shared.base import BaseModel

from ._tool_call import ChatResponseToolCall

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=PydanticBaseModel)


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
    structured_data: dict | None = Field(
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

    def as_pydantic(
        self,
        model_class: type[PydanticBaseModel] | None = None,
    ) -> PydanticBaseModel | None:
        """Convert the structured data to a pydantic model instance

        Args:
            model_class: Optional pydantic model class to use for conversion.
                         If not provided, uses the original schema class if available.

        Returns:
            Pydantic model instance or None if conversion fails
        """
        if self.structured_data is None:
            return None

        if not model_class:
            model_class = self.config.model_class_reference

        try:
            if model_class is not None:
                return model_class.model_validate(self.structured_data)
            else:
                logger.error("Error: need model_class to convert to pydantic model")
                return None
        except Exception as e:
            logger.error(f"Error converting structured data to pydantic model: {e}")
            return None

    @classmethod
    def _parse_and_validate(
        cls,
        raw_data: str | dict,
        config: StructuredOutputConfig,
    ) -> tuple[dict | None, str | None]:
        """Private helper method to parse and validate data against a schema

        Args:
            raw_data: Raw data from the model (string or dict)
            config: Configuration with schema information

        Returns:
            Tuple of (parsed_data, error_message)
        """
        error = None
        parsed_data = None

        # Get the model class from config
        model_cls: type[PydanticBaseModel] = None
        if isinstance(config.model_class_reference, type) and issubclass(
            config.model_class_reference, PydanticBaseModel
        ):
            model_cls = config.model_class_reference
        elif isinstance(config.model_class_reference, PydanticBaseModel):
            model_cls = config.model_class_reference.__class__

        # Process based on whether we have a model class and data type
        if model_cls:
            try:
                # Handle string or dict input differently
                if isinstance(raw_data, str):
                    parsed_data_pyd = model_cls.model_validate_json(raw_data)
                else:  # dict
                    parsed_data_pyd = model_cls.model_validate(raw_data)
                parsed_data = parsed_data_pyd.model_dump()
            except Exception as e:
                logger.exception(f"parse_response: Error: {e}")
                error = str(e)
        else:
            # No model class, just try to parse if string or use as is if dict
            if isinstance(raw_data, dict):
                parsed_data = raw_data
            elif isinstance(raw_data, str):
                try:
                    parsed_data = json.loads(raw_data)
                except Exception as e:
                    logger.exception(f"parse_response: Error: {e}")
                    error = str(e)
            else:
                error = f"Failed to parse response of type {type(raw_data)}"

        return parsed_data, error

    @classmethod
    def from_model_output(
        cls,
        raw_response: str | dict,
        config: StructuredOutputConfig,
    ) -> "ChatResponseStructuredOutput":
        """Create a structured output content item from model output

        Args:
            raw_response: Raw output from the model
            config: Schema to validate against

        Returns:
            ChatResponseStructuredOutput
        """
        parsed_data, error = cls._parse_and_validate(raw_response, config)

        return cls(
            config=config,
            structured_data=parsed_data,
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
        raw_response = tool_call.arguments.arguments_dict  # Get the dict directly
        parsed_data, error = cls._parse_and_validate(raw_response, config)

        return cls(
            config=config,
            structured_data=parsed_data,
            raw_data=raw_response,
            parse_error=error,
        )
