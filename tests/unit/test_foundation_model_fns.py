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
        ("gpt-5.5-pro", "open_ai"),
        ("gpt-5.4-mini", "open_ai"),
        ("gpt-5.4-nano", "open_ai"),
        ("gpt-5.3-codex", "open_ai"),
        ("claude-fable-5", "anthropic"),
        ("claude-mythos-5", "anthropic"),
        ("claude-opus-4-8", "anthropic"),
        ("claude-opus-4-7", "anthropic"),
        ("claude-sonnet-5", "anthropic"),
        ("claude-haiku-4-5", "anthropic"),
        ("gemini-3.5-flash", "google_ai"),
        ("gemini-3.1-pro-preview", "google_ai"),
        ("gemini-3.1-flash-lite", "google_ai"),
        ("deepseek-v4-flash", "deepseek"),
        ("deepseek-v4-pro", "deepseek"),
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


@pytest.mark.case_id("DAI-056")
def test_dai_056_chat_model_reasoning_controls_are_provider_neutral():
    """GIVEN registered chat foundation models
    WHEN their reasoning control metadata is inspected
    THEN only Dhenara-level reasoning controls are exposed.
    """

    allowed_controls = {"none", "effort", "token_budget"}

    for model in ALL_FOUNDATION_MODELS:
        settings = model.get_settings()
        if not hasattr(settings, "reasoning_control"):
            continue

        assert settings.reasoning_control in allowed_controls, model.model_name


@pytest.mark.case_id("DAI-057")
@pytest.mark.parametrize(
    ("model_name", "reasoning_control"),
    [
        ("gpt-5.5", "effort"),
        ("claude-fable-5", "effort"),
        ("claude-sonnet-5", "effort"),
        ("claude-opus-4-8", "effort"),
        ("claude-haiku-4-5", "token_budget"),
        ("gemini-3.5-flash", "effort"),
        ("gemini-2.5-flash", "token_budget"),
        ("deepseek-v4-pro", "effort"),
    ],
)
def test_dai_057_requested_reasoning_controls_are_generic(model_name: str, reasoning_control: str):
    """GIVEN representative reasoning models across providers
    WHEN their model settings are inspected
    THEN provider API details are normalized to Dhenara reasoning controls.
    """

    found = FoundationModelFns.get_foundation_model(model_name, all_models=ALL_FOUNDATION_MODELS)

    assert found is not None
    assert found.get_settings().reasoning_control == reasoning_control
