from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel

from ._prompt import PromptText


class SystemInstructions(BaseModel):
    instructions: list[PromptText | str] = Field(...)

    def format(self, **kwargs) -> list[str]:
        return [
            instruction.format() if isinstance(instruction, PromptText) else instruction
            for instruction in self.instructions
        ]
