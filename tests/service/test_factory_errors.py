"""Service tests for provider factory error scenarios."""

import pytest

from dhenara.ai.ai_client.factory import AIModelClientFactory
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


@pytest.mark.service
@pytest.mark.case_id("DAI-036")
def test_unsupported_provider_and_functional_type_raise(text_model):
    """
    GIVEN endpoints configured with unsupported providers or functional types
    WHEN create_provider_client is invoked
    THEN it should raise ValueError with a helpful message
    """

    api = AIModelAPI(provider=AIModelAPIProviderEnum.OPEN_AI, api_key="test")
    config = AIModelCallConfig()

    # Unsupported provider
    meta_model = FoundationModel(
        model_name="meta-llama",
        display_name="Meta Llama",
        provider=AIModelProviderEnum.META,
        functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
        settings=text_model.settings,
        valid_options=text_model.valid_options,
        cost_data=text_model.cost_data,
    )
    meta_endpoint = AIModelEndpoint(api=api, ai_model=meta_model)

    with pytest.raises(ValueError) as excinfo:
        AIModelClientFactory.create_provider_client(meta_endpoint, config, is_async=False)
    assert "Unsupported provider" in str(excinfo.value)

    # Unsupported functional type
    video_model = FoundationModel(
        model_name="video-gen",
        display_name="Video Gen",
        provider=AIModelProviderEnum.OPEN_AI,
        functional_type=AIModelFunctionalTypeEnum.VIDEO_GENERATION,
        settings=ChatModelSettings(max_input_tokens=1000, max_output_tokens=500),
        valid_options={},
        cost_data=ChatModelCostData(input_token_cost_per_million=1.0, output_token_cost_per_million=1.0),
    )
    video_endpoint = AIModelEndpoint(api=api, ai_model=video_model)

    with pytest.raises(ValueError) as excinfo_ft:
        AIModelClientFactory.create_provider_client(video_endpoint, config, is_async=False)
    assert "Unsupported functional_type" in str(excinfo_ft.value)
