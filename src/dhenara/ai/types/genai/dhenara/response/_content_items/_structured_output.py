import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from dhenara.ai.types.genai.dhenara import StructuredOutputConfig
from dhenara.ai.types.shared.base import BaseModel

from ._tool_call import ChatResponseToolCall

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=PydanticBaseModel)


def _coerce_json_strings(obj: Any) -> Any:
    """
    Recursively process data structures and convert JSON strings to Python objects.

    If a string looks like JSON (starts with { or [ and ends with } or ]),
    attempt to parse it into a Python object.
    """
    if isinstance(obj, str):
        s = obj.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return _coerce_json_strings(json.loads(s))
            except json.JSONDecodeError:
                # Not valid JSON, return the original string
                return obj
        return obj
    elif isinstance(obj, list):
        return [_coerce_json_strings(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _coerce_json_strings(v) for k, v in obj.items()}
    else:
        return obj


def _extract_json_objects_from_text(text: str) -> list[Any]:
    """Extract one or more JSON objects/arrays from a text blob.

    Robust to streaming artifacts where multiple JSON payloads are concatenated
    (e.g., "}{"), and to markdown code fences. Returns a list of parsed objects
    in the order they appear. If none are found, returns an empty list.
    """
    results: list[Any] = []
    if not isinstance(text, str):
        return results

    s = text.strip()

    # Strip markdown fences: ```json\n{...}\n```
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            body = s[nl + 1 :]
            fence_end = body.rfind("```")
            if fence_end != -1:
                s = body[:fence_end].strip()

    # Fast path: try whole string first
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = _coerce_json_strings(json.loads(s))
            results.append(obj)
            return results
        except Exception:
            pass

    # General scan: collect all balanced objects and arrays
    def _collect_balanced(open_char: str, close_char: str):
        start_idx = 0
        while True:
            start = s.find(open_char, start_idx)
            if start == -1:
                break
            depth = 0
            for i in range(start, len(s)):
                ch = s[i]
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : i + 1]
                        try:
                            parsed = _coerce_json_strings(json.loads(candidate))
                            results.append(parsed)
                        except Exception:
                            pass
                        start_idx = i + 1
                        break
            else:
                # Unbalanced - stop
                break

    _collect_balanced("{", "}")
    _collect_balanced("[", "]")
    return results


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
    post_processed: bool = Field(
        default=False,
        description="True if schema-level post-processing on error was used to coerce into the schema",
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
    ) -> tuple[dict | None, str | None, bool]:
        """Parse and validate data against schema, handling nested JSON strings"""
        error = None
        parsed_data = None
        post_processed = False

        try:
            # Step 1: Initial parsing if the input is a string
            initial_data = raw_data
            if isinstance(raw_data, str):
                try:
                    initial_data = json.loads(raw_data)
                except json.JSONDecodeError:
                    # Not valid JSON at top level, keep as string for extraction/validation
                    initial_data = raw_data

            # Step 2: Recursively normalize all nested JSON strings
            normalized_data = _coerce_json_strings(initial_data)

            # Special case: if still a string, try to extract embedded JSON blocks.
            # If multiple are found (e.g., streaming concatenation), prefer the last.
            if isinstance(normalized_data, str):
                extracted_list = _extract_json_objects_from_text(normalized_data)
                if extracted_list:
                    normalized_data = extracted_list[-1]
                    logger.warning(
                        "Structured output text contained multiple JSON payloads; using the last parsed object"
                    )

            # Step 3: Get model class from config
            model_cls: type[PydanticBaseModel] = None
            if isinstance(config.model_class_reference, type):
                try:
                    if issubclass(config.model_class_reference, PydanticBaseModel):
                        model_cls = config.model_class_reference
                except TypeError:
                    # Guard against environments where the second arg isn't a class
                    model_cls = None
            elif isinstance(config.model_class_reference, PydanticBaseModel):
                model_cls = config.model_class_reference.__class__

            # Step 4: Validate with model class if available
            if model_cls:
                try:
                    # Always use model_validate instead of model_validate_json since
                    # we've already parsed any JSON strings
                    parsed_data_pyd = model_cls.model_validate(normalized_data)
                    parsed_data = parsed_data_pyd.model_dump()
                except Exception as e:
                    logger.exception(f"Model validation error: {e}")
                    error = str(e)
                    # Attempt model-level post-processing fallback if allowed
                    allow_post = config.allow_post_process_on_error
                    if allow_post:
                        try:
                            post_fn = getattr(model_cls, "schema_post_process_on_error", None)
                            if callable(post_fn):
                                fallback_data = post_fn(normalized_data)
                                parsed_data_pyd = model_cls.model_validate(fallback_data)
                                parsed_data = parsed_data_pyd.model_dump()
                                error = None
                                post_processed = True
                        except Exception as post_e:
                            logger.warning(f"Post-process fallback failed: {post_e}")
            else:
                # No model class available, just return the normalized data
                parsed_data = normalized_data

        except Exception as e:
            logger.exception(f"Unexpected error during parsing/validation: {e}")
            error = str(e)

        return parsed_data, error, post_processed

    @classmethod
    def from_tool_call(
        cls,
        raw_response: str | dict,
        tool_call: ChatResponseToolCall | None,
        config: StructuredOutputConfig,
    ) -> "ChatResponseStructuredOutput":
        """Create a structured output from a tool call response

        Args:
            tool_call: The tool call response
            config: StructuredOutputConfig to use for validation

        Returns:
            ChatResponseStructuredOutput instance
        """

        if tool_call is not None:
            if tool_call.arguments:
                raw_response_to_parse = tool_call.arguments  # Get the dict directly
                parsed_data, error, post_processed = cls._parse_and_validate(raw_response_to_parse, config)
                # Always keep the original raw response for downstream use
                raw_data = raw_response
            else:
                parsed_data = None
                error = tool_call.parse_error
                raw_data = raw_response
                post_processed = False
        else:
            parsed_data = None
            error = "No tool call provided with `from_tool_call` method"
            raw_data = raw_response
            post_processed = False

        return cls(
            config=config,
            structured_data=parsed_data,
            raw_data=raw_data,  # Keep original response regardless of parsing
            parse_error=error,
            post_processed=post_processed,
        )
