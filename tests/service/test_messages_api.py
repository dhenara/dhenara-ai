"""Service tests for messages API round-trip behaviour."""

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types import ExternalApiCallStatusEnum
from dhenara.ai.types.genai.dhenara.request import Prompt

from ..helpers import FakeProvider


@pytest.mark.service
@pytest.mark.case_id("DAI-033")
def test_assistant_message_roundtrip_in_multi_turn(monkeypatch, text_endpoint, default_call_config):
    """
    GIVEN a conversation using the messages API
    WHEN the assistant reply is converted back into a message item for the next turn
    THEN subsequent calls should receive the accumulated history in messages
    """

    holder: dict[str, FakeProvider] = {}

    def fake_provider(cls, model_endpoint, config, is_async):
        provider = FakeProvider(model_endpoint=model_endpoint, config=config, is_async=is_async)
        holder["provider"] = provider
        return provider

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=default_call_config,
        is_async=False,
    )

    user_prompt = Prompt.with_text("Summarise the design")
    response = client.generate(messages=[user_prompt])
    assert response.status.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS
    assistant_message = response.chat_response.to_message_item() if response.chat_response else None
    assert assistant_message is not None

    conversation = [user_prompt, assistant_message]
    response_second = client.generate(messages=conversation)
    assert response_second.status.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS

    recorded_messages = holder["provider"].recorded_params.get("messages", [])
    assert len(recorded_messages) == len(conversation)
