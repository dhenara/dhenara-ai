"""Component tests for AIModelClient input validation behaviour."""

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types import ExternalApiCallStatusEnum
from dhenara.ai.types.genai.dhenara.request import Prompt

from ..helpers import make_fake_provider


@pytest.mark.component
@pytest.mark.case_id("DAI-015")
def test_input_exclusivity_and_format_toggle(monkeypatch, text_endpoint, default_call_config):
    """
    GIVEN an AIModelClient using the default formatter pipeline
    WHEN both prompt/context and messages are supplied together
    THEN the request should be rejected with REQUEST_NOT_SEND status
    AND calls with valid messages-only inputs should succeed
    """

    monkeypatch.setattr(
        AIModelClientFactory,
        "create_provider_client",
        classmethod(make_fake_provider()),
    )

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=default_call_config,
        is_async=False,
    )

    # Mixed prompt + messages must fail validation
    response = client.generate(
        prompt="Hello",
        messages=[Prompt.with_text("Hi there")],
    )

    assert response.status is not None
    assert response.status.status == ExternalApiCallStatusEnum.REQUEST_NOT_SEND
    assert "Input validation failed" in response.status.message

    # Messages-only flow should pass and return parsed response text
    ok_response = client.generate(
        prompt=None,
        messages=[Prompt.with_text("Only messages")],
    )

    assert ok_response.status is not None
    assert ok_response.status.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS
    assert ok_response.chat_response is not None
    assert ok_response.chat_response.text() == "fake-response"
