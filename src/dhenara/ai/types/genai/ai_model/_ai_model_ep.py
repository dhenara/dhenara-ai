from pydantic import Field, model_validator

from dhenara.ai.types.genai.ai_model import (
    AIModelAPI,
    BaseAIModel,
    ChatModelCostData,
    ChatResponseUsage,
    ImageModelCostData,
    ImageResponseUsage,
)
from dhenara.ai.types.shared.base import BaseModel


class AIModelEndpoint(BaseModel):
    """
    Pydantic model representing an AI model endpoint configuration.
    """

    api: AIModelAPI = Field(
        ...,
        description="Reference to API credentials",
    )
    ai_model: BaseAIModel = Field(
        ...,
        description="Reference to AI model",
    )
    order: int = Field(
        0,
        description="Order for display purposes",
    )
    enabled: bool = Field(
        True,  # noqa: FBT003
        description="Whether the endpoint is enabled",
    )
    cost_data: ChatModelCostData | ImageModelCostData | None = Field(
        None,
        description="Matching foundation model for parameter preloading",
    )
    reference_number: str | None = Field(
        None,
        description="reference number. Should be unique if not None",
    )

    @model_validator(mode="after")
    def _validate_cost_data(self) -> "AIModelEndpoint":
        if self.cost_data:
            (setting_model, cost_model) = BaseAIModel.get_pydantic_model_classes(self.functional_type)
            if not isinstance(self.cost_data, cost_model):
                raise ValueError(f"For {self.functional_type} endpoins, cost data must be type {cost_model} or None.")

        return self

    def get_cost_data(self):
        if self.cost_data:
            return self.cost_data
        else:
            return self.ai_model.get_cost_data()

    def calculate_usage_charge(
        self,
        usage: ChatResponseUsage | ImageResponseUsage,
    ):
        cost_data = self.get_cost_data()
        return cost_data.calculate_usage_charge(usage)
