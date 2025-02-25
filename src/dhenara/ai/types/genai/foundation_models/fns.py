# ruff: noqa: E501

from dhenara.ai.types.genai.ai_model import (
    FoundationModel,
)

from .constants import ALL_FOUNDATION_MODELS


class FoundationModelFns:
    @staticmethod
    def get_foundation_model(name, all_models: list[FoundationModel] | None = None):
        _all_models = all_models if all_models else ALL_FOUNDATION_MODELS
        try:
            return next(model for model in _all_models if model.model_name == name)
        except StopIteration:
            return None
