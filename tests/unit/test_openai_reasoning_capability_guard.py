import pytest

from dhenara.ai.providers.openai.responses import OpenAIResponses
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


@pytest.mark.unit
@pytest.mark.case_id("DAI-062")
def test_openai_responses_omits_reasoning_when_model_not_supported():
    model = FoundationModel(
        model_name="test-non-reasoning-model",
        display_name="Test Non-Reasoning Model",
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

    api = AIModelAPI(
        provider=AIModelAPIProviderEnum.OPEN_AI,
        api_key="test-key",
    )

    endpoint = AIModelEndpoint(
        api=api,
        ai_model=model,
    )

    cfg = AIModelCallConfig(reasoning=True, reasoning_effort="high")

    client = OpenAIResponses(model_endpoint=endpoint, config=cfg, is_async=False)
    client._client = object()  # bypass initialization requirement
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt=None, context=[], instructions=None, messages=[])
    args = params["response_args"]

    assert "reasoning" not in args
