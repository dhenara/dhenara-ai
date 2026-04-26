from __future__ import annotations

import pytest

from dhenara.ai.providers.google.chat import GoogleAIChat
from dhenara.ai.providers.shared.shared import APIProviderSharedFns
from dhenara.ai.types.genai import AIModelAPI, AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum
from dhenara.ai.types.genai.foundation_models.google.chat import (
    Gemini25Pro,
    Gemini3FlashPreview,
    Gemini31FlashLitePreview,
    Gemini31ProPreview,
)

pytestmark = [pytest.mark.unit]


def _build_vertex_client(ai_model: object, configured_location: str = "us-central1") -> GoogleAIChat:
    api = AIModelAPI(
        provider=AIModelAPIProviderEnum.GOOGLE_VERTEX_AI,
        credentials={"service_account_json": {"project_id": "svc-project"}},
        config={"project_id": "test-project", "location": configured_location},
    )
    endpoint = AIModelEndpoint(api=api, ai_model=ai_model)
    return GoogleAIChat(model_endpoint=endpoint, config=AIModelCallConfig(), is_async=False)


@pytest.mark.case_id("DAI-301")
@pytest.mark.parametrize(
    "ai_model",
    [
        Gemini31ProPreview,
        Gemini3FlashPreview,
        Gemini31FlashLitePreview,
    ],
)
def test_dai_301_google_vertex_uses_global_location_for_models_marked_global_only(
    monkeypatch: pytest.MonkeyPatch,
    ai_model: object,
) -> None:
    monkeypatch.setattr(
        APIProviderSharedFns,
        "get_vertex_ai_credentials",
        staticmethod(
            lambda _api: {
                "credentials": object(),
                "project_id": "test-project",
                "location": "us-central1",
            }
        ),
    )

    client = _build_vertex_client(ai_model)
    client_type, params = client._get_client_params(client.model_endpoint.api)

    assert client_type == "vertex_ai"
    assert params["location"] == "global"


@pytest.mark.case_id("DAI-302")
def test_dai_302_google_vertex_beta_flag_alone_does_not_force_global_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        APIProviderSharedFns,
        "get_vertex_ai_credentials",
        staticmethod(
            lambda _api: {
                "credentials": object(),
                "project_id": "test-project",
                "location": "us-central1",
            }
        ),
    )

    beta_regional_model = Gemini25Pro.create_instance(
        model_name="gemini-2.5-pro-beta-test",
        display_name="Gemini 2.5 Pro Beta Test",
        beta=True,
    )
    client = _build_vertex_client(beta_regional_model)
    _client_type, params = client._get_client_params(client.model_endpoint.api)

    assert params["location"] == "us-central1"


@pytest.mark.case_id("DAI-303")
def test_dai_303_google_vertex_keeps_regional_location_for_models_with_regional_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        APIProviderSharedFns,
        "get_vertex_ai_credentials",
        staticmethod(
            lambda _api: {
                "credentials": object(),
                "project_id": "test-project",
                "location": "us-central1",
            }
        ),
    )

    client = _build_vertex_client(Gemini25Pro)
    _client_type, params = client._get_client_params(client.model_endpoint.api)

    assert params["location"] == "us-central1"
