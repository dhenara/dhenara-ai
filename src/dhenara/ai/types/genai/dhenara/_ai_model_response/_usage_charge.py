
from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel


class UsageCharge(BaseModel):
    cost: float = Field(
        ...,
        description="Cost",
    )
    charge: float | None = Field(
        ...,
        description="Charge after considering internal expences and margins."
        " Will be  None if  `cost_multiplier_percentage` is not set in cost data.",
    )
