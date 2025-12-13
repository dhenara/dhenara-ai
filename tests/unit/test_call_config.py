"""Unit tests for AIModelCallConfig class.

Tests cover max_output_tokens and reasoning configuration behavior.
"""

import pytest

from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)
from dhenara.ai.types.genai.dhenara.request import AIModelCallConfig


@pytest.mark.unit
@pytest.mark.case_id("DAI-003")
def test_get_max_output_tokens_non_reasoning_cap():
    """
    GIVEN an AIModelCallConfig with reasoning=False and max_output_tokens=5000
    WHEN get_max_output_tokens is called with a model that has max_output_tokens=4096
    THEN the returned max_output_tokens should be capped to 4096 (model limit)
    AND max_reasoning_tokens should be None
    """
    # Create a test model with specific limits
    model = FoundationModel(
        model_name="test-model",
        display_name="Test Model",
        provider=AIModelProviderEnum.OPEN_AI,
        functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
        settings=ChatModelSettings(
            max_input_tokens=100000,
            max_output_tokens=4096,
            supports_reasoning=False,
        ),
        valid_options={},
        cost_data=ChatModelCostData(
            input_token_cost_per_million=5.0,
            output_token_cost_per_million=15.0,
        ),
    )

    # Create config with higher limit than model supports
    config = AIModelCallConfig(
        reasoning=False,
        max_output_tokens=5000,
    )

    max_output, max_reasoning = config.get_max_output_tokens(model)

    # Should be capped to model's limit
    assert max_output == 4096
    assert max_reasoning is None


@pytest.mark.unit
@pytest.mark.case_id("DAI-004")
def test_get_max_tokens_reasoning_with_override():
    """
    GIVEN an AIModelCallConfig with reasoning=True, max_output_tokens=2000, max_reasoning_tokens=3000
    WHEN get_max_output_tokens is called with a reasoning-capable model
    THEN both max_output_tokens and max_reasoning_tokens should respect model limits and config overrides
    """
    # Create a reasoning-capable model
    # Note: max_reasoning_tokens must be <= max_output_tokens_reasoning_mode
    model = FoundationModel(
        model_name="test-reasoning-model",
        display_name="Test Reasoning Model",
        provider=AIModelProviderEnum.OPEN_AI,
        functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
        settings=ChatModelSettings(
            max_input_tokens=100000,
            max_output_tokens=4096,
            max_output_tokens_reasoning_mode=12000,
            max_reasoning_tokens=10000,  # Must be <= 12000
            supports_reasoning=True,
        ),
        valid_options={},
        cost_data=ChatModelCostData(
            input_token_cost_per_million=5.0,
            output_token_cost_per_million=15.0,
        ),
    )

    # Test 1: Config values lower than model limits
    config = AIModelCallConfig(
        reasoning=True,
        max_output_tokens=2000,
        max_reasoning_tokens=3000,
    )
    max_output, max_reasoning = config.get_max_output_tokens(model)
    assert max_output == 2000  # Uses config value
    assert max_reasoning == 3000  # Uses config value

    # Test 2: Config values higher than model limits
    config2 = AIModelCallConfig(
        reasoning=True,
        max_output_tokens=15000,  # Higher than model's 12000
        max_reasoning_tokens=15000,  # Higher than model's 10000
    )
    max_output2, max_reasoning2 = config2.get_max_output_tokens(model)
    assert max_output2 == 12000  # Capped to model limit (max_output_tokens_reasoning_mode)
    assert max_reasoning2 == 10000  # Capped to model limit

    # Test 3: No config overrides (use model defaults)
    config3 = AIModelCallConfig(reasoning=True)
    max_output3, max_reasoning3 = config3.get_max_output_tokens(model)
    assert max_output3 == 12000  # Model's reasoning mode limit
    assert max_reasoning3 == 10000  # Model's reasoning limit


@pytest.mark.unit
@pytest.mark.case_id("DAI-006")
def test_reasoning_model_without_reasoning_mode_tokens_falls_back():
    """GIVEN a model that supports_reasoning=True but has no max_output_tokens_reasoning_mode
    WHEN reasoning=True in the call-config
    THEN get_max_output_tokens should fall back to max_output_tokens (no error)
    AND max_reasoning_tokens should be None if the model doesn't define it.
    """

    model = FoundationModel(
        model_name="test-reasoning-model-missing-reasoning-mode-max",
        display_name="Test Reasoning Model Missing Reasoning Mode Max",
        provider=AIModelProviderEnum.OPEN_AI,
        functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
        settings=ChatModelSettings(
            max_input_tokens=100000,
            max_output_tokens=4096,
            supports_reasoning=True,
            # Intentionally omit: max_output_tokens_reasoning_mode
            # Intentionally omit: max_reasoning_tokens
        ),
        valid_options={},
        cost_data=ChatModelCostData(
            input_token_cost_per_million=5.0,
            output_token_cost_per_million=15.0,
        ),
    )

    config = AIModelCallConfig(
        reasoning=True,
        max_output_tokens=5000,
        max_reasoning_tokens=2000,
    )

    max_output, max_reasoning = config.get_max_output_tokens(model)

    assert max_output == 4096
    assert max_reasoning is None


@pytest.mark.unit
@pytest.mark.case_id("DAI-005")
def test_reasoning_effort_minimal_maps_low():
    """
    GIVEN an AIModelCallConfig with reasoning_effort="minimal"
    WHEN the config is created
    THEN the reasoning_effort field should be set to "minimal"
    AND it should be a valid literal value that providers can normalize
    """
    config = AIModelCallConfig(
        reasoning=True,
        reasoning_effort="minimal",
    )

    assert config.reasoning_effort == "minimal"
    assert config.reasoning is True

    # Test all valid reasoning effort values
    for effort in ["minimal", "low", "medium", "high"]:
        config = AIModelCallConfig(reasoning=True, reasoning_effort=effort)
        assert config.reasoning_effort == effort
