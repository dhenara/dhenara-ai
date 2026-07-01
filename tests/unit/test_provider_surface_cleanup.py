from __future__ import annotations

import pytest
from pydantic import ValidationError

from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.providers.openai.base import _normalize_microsoft_openai_base_url
from dhenara.ai.providers.openai.chat_completions_api.chat import OpenAIChatCompletions
from dhenara.ai.providers.openai.responses import OpenAIResponses
from dhenara.ai.types.genai.ai_model import AIModelAPI, AIModelAPIProviderEnum, AIModelEndpoint
from dhenara.ai.types.genai.dhenara.request import AIModelCallConfig
from dhenara.ai.types.genai.foundation_models.deepseek.chat import DeepseekV4Flash
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT5Nano
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
        models=[DeepseekV4Flash],
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


@pytest.mark.case_id("DAI-311")
def test_dai_311_deepseek_endpoint_prefers_deepseek_api_provider() -> None:
    resource_config = ResourceConfig(
        models=[DeepseekV4Flash],
        model_apis=[
            AIModelAPI(
                provider=AIModelAPIProviderEnum.MICROSOFT_OPENAI,
                api_key="12345678",
                config={"endpoint": "https://example.openai.azure.com"},
            ),
            AIModelAPI(
                provider=AIModelAPIProviderEnum.DEEPSEEK,
                api_key="12345678",
            ),
        ],
    )

    resource_config._initialize_endpoints()

    assert len(resource_config.model_endpoints) == 1
    assert resource_config.model_endpoints[0].api.provider == AIModelAPIProviderEnum.DEEPSEEK


@pytest.mark.case_id("DAI-312")
def test_dai_312_deepseek_uses_chat_completions_client() -> None:
    resource_config = ResourceConfig(
        models=[DeepseekV4Flash],
        model_apis=[
            AIModelAPI(
                provider=AIModelAPIProviderEnum.DEEPSEEK,
                api_key="12345678",
            ),
        ],
    )
    resource_config._initialize_endpoints()

    provider = AIModelClientFactory.create_provider_client(
        resource_config.model_endpoints[0],
        AIModelCallConfig(),
        is_async=False,
    )

    assert isinstance(provider, OpenAIChatCompletions)
    client_type, client_params = provider._get_client_params(resource_config.model_endpoints[0].api)
    assert client_type == "openai"
    assert client_params["base_url"] == "https://api.deepseek.com"


@pytest.mark.case_id("DAI-313")
def test_dai_313_deepseek_reasoning_false_disables_thinking() -> None:
    endpoint = AIModelEndpoint(
        api=AIModelAPI(
            provider=AIModelAPIProviderEnum.DEEPSEEK,
            api_key="12345678",
        ),
        ai_model=DeepseekV4Flash,
    )
    client = OpenAIChatCompletions(
        model_endpoint=endpoint,
        config=AIModelCallConfig(reasoning=False),
        is_async=False,
    )
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})

    assert params["chat_args"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "max_tokens" in params["chat_args"]
    assert "max_completion_tokens" not in params["chat_args"]


@pytest.mark.case_id("DAI-314")
def test_dai_314_deepseek_reasoning_effort_uses_thinking_extra_body() -> None:
    endpoint = AIModelEndpoint(
        api=AIModelAPI(
            provider=AIModelAPIProviderEnum.DEEPSEEK,
            api_key="12345678",
        ),
        ai_model=DeepseekV4Flash,
    )
    client = OpenAIChatCompletions(
        model_endpoint=endpoint,
        config=AIModelCallConfig(reasoning=True, reasoning_effort="xhigh"),
        is_async=False,
    )
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})

    assert params["chat_args"]["extra_body"] == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }


@pytest.mark.case_id("DAI-315")
def test_dai_315_deepseek_microsoft_fallback_does_not_send_direct_api_thinking_body() -> None:
    endpoint = AIModelEndpoint(
        api=AIModelAPI(
            provider=AIModelAPIProviderEnum.MICROSOFT_OPENAI,
            api_key="12345678",
            config={"endpoint": "https://example.openai.azure.com"},
        ),
        ai_model=DeepseekV4Flash,
    )
    client = OpenAIChatCompletions(
        model_endpoint=endpoint,
        config=AIModelCallConfig(reasoning=True, reasoning_effort="xhigh"),
        is_async=False,
    )
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": "hi"})

    assert params["chat_args"]["max_tokens"] == 384000
    assert "extra_body" not in params["chat_args"]


@pytest.mark.case_id("DAI-310")
def test_dai_310_openai_responses_accepts_microsoft_openai_provider() -> None:
    endpoint = AIModelEndpoint(
        api=AIModelAPI(
            provider=AIModelAPIProviderEnum.MICROSOFT_OPENAI,
            api_key="12345678",
            config={"endpoint": "https://example.openai.azure.com"},
        ),
        ai_model=GPT5Nano,
    )
    client = OpenAIResponses(model_endpoint=endpoint, config=AIModelCallConfig(), is_async=False)
    client._client = object()
    client._input_validation_pending = False

    params = client.get_api_call_params(prompt={"role": "user", "content": [{"type": "input_text", "text": "hi"}]})

    assert params["response_args"]["model"] == "gpt-5-nano"
