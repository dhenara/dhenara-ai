from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from dhenara.ai.providers.openai.base import _normalize_microsoft_openai_base_url
from dhenara.ai.providers.openai.responses import OpenAIResponses
from dhenara.ai.types.genai.ai_model import AIModelAPI, AIModelAPIProviderEnum
from dhenara.ai.types.genai.foundation_models.deepseek.chat import DeepseekR1
from dhenara.ai.types.resource import ResourceConfig

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-307")
def test_dai_307_microsoft_openai_base_url_normalization() -> None:
    assert _normalize_microsoft_openai_base_url("https://example.openai.azure.com") == (
        "https://example.openai.azure.com/openai/v1/"
    )
    assert _normalize_microsoft_openai_base_url("https://example.services.ai.azure.com/models") == (
        "https://example.services.ai.azure.com/openai/v1/"
    )
    assert _normalize_microsoft_openai_base_url("https://example.openai.azure.com/openai/v1") == (
        "https://example.openai.azure.com/openai/v1/"
    )
    assert _normalize_microsoft_openai_base_url("https://example.services.ai.azure.com/openai/v1/") == (
        "https://example.services.ai.azure.com/openai/v1/"
    )


@pytest.mark.case_id("DAI-308")
def test_dai_308_microsoft_azure_ai_is_rejected() -> None:
    with pytest.raises(ValidationError, match="microsoft_azure_ai is no longer supported"):
        AIModelAPI(
            provider=AIModelAPIProviderEnum.MICROSOFT_AZURE_AI,
            api_key="12345678",
            config={"endpoint": "https://example.models.ai.azure.com"},
        )


@pytest.mark.case_id("DAI-309")
def test_dai_309_deepseek_endpoint_falls_back_to_microsoft_openai() -> None:
    resource_config = ResourceConfig(
        models=[DeepseekR1],
        model_apis=[
            AIModelAPI(
                provider=AIModelAPIProviderEnum.MICROSOFT_OPENAI,
                api_key="12345678",
                config={"endpoint": "https://example.openai.azure.com"},
            )
        ],
    )

    resource_config._initialize_endpoints()

    assert len(resource_config.model_endpoints) == 1
    assert resource_config.model_endpoints[0].api.provider == AIModelAPIProviderEnum.MICROSOFT_OPENAI


@pytest.mark.case_id("DAI-310")
def test_dai_310_openai_responses_accepts_microsoft_openai_provider() -> None:
    client = OpenAIResponses.__new__(OpenAIResponses)
    client._client = object()
    client._input_validation_pending = False
    client.model_name_in_api_calls = "gpt-5-nano"
    client.model_endpoint = SimpleNamespace(
        api=SimpleNamespace(provider=AIModelAPIProviderEnum.MICROSOFT_OPENAI),
        ai_model=SimpleNamespace(get_settings=lambda: SimpleNamespace(supports_reasoning=False)),
    )
    client.config = SimpleNamespace(
        streaming=False,
        reasoning=False,
        reasoning_effort=None,
        tools=None,
        tool_choice=None,
        options={},
        get_max_output_tokens=lambda ai_model: (None, None),
        get_user=lambda: None,
    )
    client._get_structured_output_config = lambda: None

    params = client.get_api_call_params(prompt={"role": "user", "content": [{"type": "input_text", "text": "hi"}]})

    assert params["response_args"]["model"] == "gpt-5-nano"
