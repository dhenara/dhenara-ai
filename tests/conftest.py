import pytest

from dhenara.ai.types.genai.ai_model import (
    AIModelAPI,
    AIModelAPIProviderEnum,
    AIModelEndpoint,
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    FoundationModel,
)
from dhenara.ai.types.genai.dhenara.request import AIModelCallConfig


@pytest.fixture(scope="session")
def package_name() -> str:
    """Marker fixture to identify the package under test."""
    return "dhenara-ai"


@pytest.fixture()
def text_model() -> FoundationModel:
    """Create a baseline text generation foundation model for tests."""

    return FoundationModel(
        model_name="test-model",
        display_name="Test Model",
        provider=AIModelProviderEnum.OPEN_AI,
        functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
        settings=ChatModelSettings(
            max_input_tokens=100000,
            max_output_tokens=4096,
            max_output_tokens_reasoning_mode=8192,
            max_reasoning_tokens=2048,
            supports_reasoning=True,
        ),
        valid_options={},
        cost_data=ChatModelCostData(
            input_token_cost_per_million=5.0,
            output_token_cost_per_million=15.0,
        ),
    )


@pytest.fixture()
def text_endpoint(text_model: FoundationModel) -> AIModelEndpoint:
    """Create an AIModelEndpoint backed by the baseline text model."""

    api = AIModelAPI(
        provider=AIModelAPIProviderEnum.OPEN_AI,
        api_key="test-key",
    )
    return AIModelEndpoint(api=api, ai_model=text_model)


@pytest.fixture()
def default_call_config() -> AIModelCallConfig:
    """Return a baseline call config used across tests."""

    return AIModelCallConfig(
        streaming=False,
        reasoning=False,
        retries=2,
        retry_delay=0.01,
        max_retry_delay=0.05,
        timeout=0.1,
    )
