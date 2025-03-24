from pydantic import Field, model_validator

from dhenara.ai.types.shared.base import BaseModel

from ._content import Content
from ._role import PromptMessageRoleEnum
from ._text_template import TextTemplate


class PromptText(BaseModel):
    content: Content | None = Field(
        default=None,
        description="Prompt Content",
    )
    template: TextTemplate | None = Field(
        default=None,
        description="Text template with optional {placeholders} for string formatting",
    )

    @model_validator(mode="after")
    def validate_all(self) -> "PromptText":
        if not (self.content or self.template):
            raise ValueError("Content or Template is required for prompt")
        if self.content and self.template:
            raise ValueError("Only one of Content or Template is allowed")
        return self

    def format(self, **kwargs) -> str:
        if self.content:
            return self.content.get_content()
        else:
            return self.template.format(**kwargs)


class Prompt(BaseModel):
    role: PromptMessageRoleEnum
    text: str | PromptText

    def format(self, **kwargs) -> dict:
        """Format the prompt as a generic dictionary"""
        if isinstance(self.text, PromptText):
            formatted_text = self.text.format(**kwargs)
        else:
            formatted_text = self.text

        return {"role": self.role.value, "content": formatted_text}


class FormattedPrompt(BaseModel):
    role: PromptMessageRoleEnum
    text: str

    ''' TODO: Delete
    def to_openai_format(self) -> dict:
        """Convert to OpenAI message format"""
        result = self.format()
        # OpenAI already uses role:content format
        return result

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic message format"""
        result = self.format()
        # Map generic role to Anthropic-specific role if needed
        role_mapping = {
            "user": "user",
            "assistant": "assistant",
            "system": "assistant",  # Anthropic doesn't have system role in messages
        }
        result["role"] = role_mapping.get(result["role"], result["role"])
        return result

    def to_google_format(self) -> dict:
        """Convert to Google message format"""
        formatted = self.format()
        # Google AI uses "parts" instead of "content"
        role_mapping = {
            "user": "user",
            "assistant": "model",
            "system": "user",  # Google doesn't have system role, use as user
        }
        return {
            "role": role_mapping.get(formatted["role"], formatted["role"]),
            "parts": [{"text": formatted["content"]}],
        }

    @classmethod
    def from_openai_format(cls, openai_message: dict) -> "Prompt":
        """Create Prompt from OpenAI message format"""
        role = openai_message.get("role", "user")
        if isinstance(openai_message.get("content"), list):
            # Handle content as list (multimodal)
            text_parts = []
            for item in openai_message["content"]:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            text = " ".join(text_parts)
        else:
            text = openai_message.get("content", "")

        return cls(role=PromptMessageRoleEnum(role), text=text)

    @classmethod
    def from_anthropic_format(cls, anthropic_message: dict) -> "Prompt":
        """Create Prompt from Anthropic message format"""
        role = anthropic_message.get("role", "user")
        # Map Anthropic roles to our roles
        role_mapping = {"user": "user", "assistant": "assistant"}
        mapped_role = role_mapping.get(role, "user")

        if isinstance(anthropic_message.get("content"), list):
            # Handle content as list
            text_parts = []
            for item in anthropic_message["content"]:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            text = " ".join(text_parts)
        else:
            text = anthropic_message.get("content", "")

        return cls(role=PromptMessageRoleEnum(mapped_role), text=text)

    @classmethod
    def from_google_format(cls, google_message: dict) -> "Prompt":
        """Create Prompt from Google message format"""
        role = google_message.get("role", "user")
        # Map Google roles to our roles
        role_mapping = {"user": "user", "model": "assistant"}
        mapped_role = role_mapping.get(role, "user")

        # Google uses "parts" instead of "content"
        parts = google_message.get("parts", [])
        text_parts = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)

        text = " ".join(text_parts)

        return cls(role=PromptMessageRoleEnum(mapped_role), text=text)

    '''
