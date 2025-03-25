import json
from typing import Any

# from urllib.request import urlopen
from pydantic import Field

from dhenara.ai.types.shared.base import BaseEnum, BaseModel


class ContentType(BaseEnum):
    """Enumeration of content types that can be returned."""

    TEXT = "text"
    LIST = "list"  # Text List
    DICT = "dict"  # Json
    JSONL = "jsonl"


class Content(BaseModel):
    """Represents user input data for AI model processing.

    This model handles various forms of input content including text, URLs, and JSON data
    that can be processed by AI models.

    """

    type: ContentType = Field(
        default=None,
    )
    text: str | None = Field(
        default=None,
        description="Primary text content to be processed",
        example="What is the capital of France?",
    )
    textl: list[str] | None = Field(
        default=None,
        description="Multiple text contents for batch processing",
        example=["Text 1", "Text 2"],
    )

    # Not using name `json` as it shadows an attribute in parent "BaseModel"
    json_c: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured JSON data for processing",
        example={"key": "value"},
    )
    jsonl_c: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of JSON objects in JSONL format",
        example=[{"id": 1, "text": "example"}, {"id": 2, "text": "example2"}],
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "examples": [
                {
                    "text": "What is the capital of France?",
                    "textl": ["Text 1", "Text 2"],
                    "json_c": {"key": "value"},
                    "jsonl_c": [{"id": 1, "text": "example"}],
                },
            ],
        }

    @property
    def has_content(self) -> bool:
        """Check if the input contains any content.

        Returns:
            bool: True if any content field is populated, False otherwise.
        """
        return any(
            [
                self.text is not None,
                self.textl is not None,
                bool(self.json_c),
                bool(self.jsonl_c),
            ]
        )

    # def validate_content_length(self, content: str) -> str:
    #    if len(content) > 8192:
    #        raise ValueError("Content exceeds maximum length of 8192 characters")
    #    return content

    def get_content(
        self,
        separator: str = "\n",
    ) -> str | list[str] | dict | list[dict]:
        """Retrieve content from any of the input sources.

        This method consolidates content from various input fields and returns it in the
        requested format. It handles text content, URLs, JSON, and JSONL data.

        Args:
            return_type: Desired return type format (text, list, dict, or jsonl_c)
            separator: String separator for joining text content when return_type is TEXT

        Returns:
            Content in the requested format:
            - TEXT: Single string with all content joined
            - LIST: List of strings
            - DICT: Dictionary from json_c
            - JSONL: List of dictionaries from jsonl_c
        """
        if not self.has_content:
            raise ValueError("No content available in any field")

        # Collect all text content
        text_list: list[str] = []

        # Add single content if present
        if self.text:
            text_list.append(self.text)

        # Add multiple textl if present
        if self.textl:
            text_list.extend(self.textl)

        # Return based on requested type
        if self.type == ContentType.TEXT:
            # Include JSON content if present
            if self.json_c:
                text_list.append(json.dumps(self.json_c, ensure_ascii=False))
            if self.jsonl_c:
                text_list.extend(json.dumps(item, ensure_ascii=False) for item in self.jsonl_c)
            return separator.join(text_list)

        elif self.type == ContentType.LIST:
            return text_list

        elif self.type == ContentType.DICT:
            if not self.json_c:
                raise ValueError("No JSON content available")
            return self.json_c

        elif self.type == ContentType.JSONL:
            if not self.jsonl_c:
                raise ValueError("No JSONL content available")
            return self.jsonl_c
        else:
            raise ValueError(f"get_content: Unknown type {self.type}")
