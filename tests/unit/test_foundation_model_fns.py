# Provenance: Added to improve foundation model helpers coverage (2026-01-21)

from __future__ import annotations

import pytest

from dhenara.ai.types.genai.foundation_models import ALL_FOUNDATION_MODELS
from dhenara.ai.types.genai.foundation_models.fns import FoundationModelFns

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-054")
def test_dai_054_get_foundation_model_by_name_and_missing():
    """GIVEN a list of foundation models
    WHEN get_foundation_model is called with an existing and missing name
    THEN it returns the model for existing name and None otherwise.
    """

    assert ALL_FOUNDATION_MODELS
    name = ALL_FOUNDATION_MODELS[0].model_name

    found = FoundationModelFns.get_foundation_model(name, all_models=ALL_FOUNDATION_MODELS)
    assert found is not None
    assert found.model_name == name

    assert FoundationModelFns.get_foundation_model("definitely-not-a-model", all_models=ALL_FOUNDATION_MODELS) is None


@pytest.mark.case_id("DAI-055")
@pytest.mark.parametrize(
    ("model_name", "provider"),
    [
        ("gpt-5.5", "open_ai"),
        ("gpt-5.4-mini", "open_ai"),
        ("gpt-5.4-nano", "open_ai"),
        ("claude-opus-4-7", "anthropic"),
        ("claude-haiku-4-5", "anthropic"),
        ("gemini-3.1-pro-preview", "google_ai"),
    ],
)
def test_dai_055_requested_models_are_registered(model_name: str, provider: str):
    """GIVEN requested foundation model names
    WHEN the registry is queried by exact model name
    THEN the matching model is present with the expected provider.
    """

    found = FoundationModelFns.get_foundation_model(model_name, all_models=ALL_FOUNDATION_MODELS)

    assert found is not None
    assert found.model_name == model_name
    assert found.provider == provider
