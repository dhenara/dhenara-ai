"""Service tests covering tool call round-trip behaviour."""

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types import ExternalApiCallStatusEnum
from dhenara.ai.types.genai import (
    ChatResponse,
    ChatResponseChoice,
    ChatResponseToolCall,
    ChatResponseToolCallContentItem,
)
from dhenara.ai.types.genai.dhenara.request import Prompt
from dhenara.ai.types.genai.dhenara.response import AIModelCallResponseMetaData

from ..helpers import FakeProvider


class _ToolCallProvider(FakeProvider):
    """Provider that emits a tool call in the response content."""

    def parse_response(self, response):  # type: ignore[override]
        tool_content = ChatResponseToolCallContentItem(
            index=0,
            role="assistant",
            tool_call=ChatResponseToolCall(
                call_id="call-1",
                id="tool-1",
                name="fetch_signal",
                arguments={"path": "/tmp/log"},
                metadata={"source": "unit-test"},
            ),
            metadata={"confidence": 0.9},
        )
        choice = ChatResponseChoice(index=0, contents=[tool_content])
        return ChatResponse(
            model=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            choices=[choice],
            metadata=AIModelCallResponseMetaData(streaming=False, duration_seconds=0, provider_metadata={}),
            provider_response=response,
        )


@pytest.mark.service
@pytest.mark.case_id("DAI-034")
def test_tool_call_and_result_message_flow(monkeypatch, text_endpoint, default_call_config):
    """
    GIVEN a provider response containing a tool call
    WHEN converted to a message item for the next turn
    THEN the tool call payload should be preserved for downstream execution
    """

    def fake_provider(cls, model_endpoint, config, is_async):
        return _ToolCallProvider(model_endpoint=model_endpoint, config=config, is_async=is_async)

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=default_call_config,
        is_async=False,
    )

    response = client.generate(messages=[Prompt.with_text("Invoke tooling")])
    assert response.status.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS

    assistant_message = response.chat_response.to_message_item() if response.chat_response else None
    assert assistant_message is not None
    tools = assistant_message.tools() if assistant_message else []
    assert tools and tools[0].name == "fetch_signal"
    assert tools[0].arguments["path"] == "/tmp/log"
