from typing import Any

from pydantic import Field, model_validator

from dhenara.ai.types.shared.base import BaseModel
from dhenara.ai.types.shared.file import GenericFile

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


class PromptConfig(BaseModel):
    max_words_text: int | None = Field(
        default=None,
        description="Max number of words in text. Leave as None for not limiting.",
    )
    max_words_file: int | None = Field(
        default=None,
        description="Max number of words in a single file. Leave as None for not limiting.",
    )


class BaseTextPrompt(BaseModel):
    text: str | PromptText = Field(
        ...,
        description="Prompt Text.",
    )
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Variable name s and values for template resolution in prompt.",
        example={"style": "modern", "name": "Annie"},
    )

    def get_formatted_text(
        self,
        max_words: int | None = None,
        **kwargs,
    ) -> str:
        """Format the prompt as a generic dictionary"""
        if isinstance(self.text, PromptText):
            var_dict = self.variables.copy()
            var_dict.update(**kwargs)
            formatted_text = self.text.format(**var_dict)
        elif isinstance(self.text, str):
            formatted_text = self.text
        else:
            raise ValueError(f"get_formatted_text: unknown prompt.text type {type(self.text)}")

        if max_words:
            words = formatted_text.split()
            formatted_text = " ".join(words[:max_words])

        return formatted_text


class Prompt(BaseTextPrompt):
    role: PromptMessageRoleEnum = Field(
        ...,
        description="Role",
    )
    files: list[GenericFile] = Field(
        default_factory=list,
        description="Files",
    )
    config: PromptConfig | None = Field(
        default=None,
        description="Prompt Config.",
    )

    @classmethod
    def with_text(
        cls,
        text: str,
        variables: dict | None = None,
        disable_checks: bool = False,
    ):
        return cls(
            role=PromptMessageRoleEnum.USER,
            text=PromptText(
                content=None,
                template=TextTemplate(
                    text=text,
                    variables=variables or {},
                    disable_checks=disable_checks,
                ),
            ),
        )

    @classmethod
    def with_dad_text(
        cls,
        text: str,
        variables: dict | None = None,
        disable_checks: bool = True,
    ):
        return cls(
            role=PromptMessageRoleEnum.USER,
            text=PromptText(
                content=None,
                template=TextTemplate(
                    text=text,
                    variables=variables or {},
                    disable_checks=disable_checks,
                ),
            ),
        )


class FormattedPrompt(BaseModel):
    role: PromptMessageRoleEnum
    text: str


class SystemInstruction(BaseTextPrompt):
    pass
