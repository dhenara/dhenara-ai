from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dhenara.ai.types import AIModelAPIProviderEnum, AIModelEndpoint
from dhenara.ai.types.genai.dhenara.request import Prompt, ToolCallResult
from dhenara.ai.types.genai.dhenara.response import ChatResponse

from ._config import provider_model_cases, select_provider_endpoint
from ._helpers import (
    ArtifactTracker,
    run_async_streaming_multi_turn,
    run_function_calling,
    run_image_generation,
    run_messages_api,
    run_messages_streaming,
    run_multi_turn_conversation,
    run_streaming_multi_turn,
    run_streaming_tools_structured,
    run_structured_output_all_providers,
    run_structured_output_messages,
    run_structured_output_single_turn,
    run_structured_thinking,
    run_text_generation_sync,
    run_text_streaming_sync,
    run_tools_with_messages,
    run_various_input_formats,
)

pytestmark = pytest.mark.realtime


@dataclass(frozen=True)
class ProviderModelUnderTest:
    provider: AIModelAPIProviderEnum
    model_name: str
    endpoint: AIModelEndpoint

    @property
    def slug(self) -> str:
        provider_slug = self.provider.name.lower()
        model_slug = self.model_name.replace(".", "-")
        return f"{provider_slug}-{model_slug}"


@pytest.fixture(scope="session", params=provider_model_cases(), ids=lambda data: f"{data[0].name.lower()}-{data[1]}")
def provider_model_endpoint(request, realtime_resource_config):
    provider, model_name = request.param
    endpoint = select_provider_endpoint(
        realtime_resource_config,
        provider,
        preferred_models=(model_name,),
    )
    return ProviderModelUnderTest(provider=provider, model_name=model_name, endpoint=endpoint)


def test_realtime_text_generation(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we run the text-generation scenario
    THEN the response contains at least one assistant choice
    """

    response = run_text_generation_sync(provider_model_endpoint.endpoint)
    assert response.choices


def test_realtime_streaming_text(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we trigger the streaming scenario
    THEN token deltas are produced for the response
    """

    stream = run_text_streaming_sync(provider_model_endpoint.endpoint)
    assert stream.deltas


def test_realtime_input_formats(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we submit prompts using multiple payload formats
    THEN each representation yields a non-empty textual response
    """

    outputs = run_various_input_formats(provider_model_endpoint.endpoint)
    assert len(outputs) == 4


def test_realtime_multi_turn(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we drive the multi-turn conversation scenario
    THEN three conversation nodes are captured in history
    """

    history = run_multi_turn_conversation(provider_model_endpoint.endpoint)
    assert len(history) == 3


def test_realtime_streaming_multi_turn(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we run the streaming multi-turn conversation
    THEN two streamed responses are appended to history
    """

    history = run_streaming_multi_turn(provider_model_endpoint.endpoint)
    assert len(history) == 2


@pytest.mark.asyncio
async def test_realtime_async_streaming_multi_turn(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we await the async streaming multi-turn run
    THEN two streamed responses are collected
    """

    history = await run_async_streaming_multi_turn(provider_model_endpoint.endpoint)
    assert len(history) == 2


def test_realtime_messages_api(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we exercise the messages API flow
    THEN the scenario returns an ordered transcript
    """

    messages = run_messages_api(provider_model_endpoint.endpoint)
    assert messages


def test_realtime_streaming_messages(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we stream assistant replies via the messages API
    THEN at least one assistant message is emitted
    """

    messages = run_messages_streaming(provider_model_endpoint.endpoint)
    assert messages


def test_realtime_tools_with_messages(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we allow the assistant to trigger tools mid-dialog
    THEN the transcript captures the intermediate tool calls
    """

    messages = run_tools_with_messages(provider_model_endpoint.endpoint)
    assert messages


def test_realtime_structured_output_messages(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we request structured analysis via messages
    THEN the structured payload contains a summary field
    """

    structured = run_structured_output_messages(provider_model_endpoint.endpoint)
    assert structured.summary


def test_realtime_streaming_tools_structured(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we stream tool-led reasoning with structured output
    THEN textual output is returned and structured weather data is optional
    """

    text, structured = run_streaming_tools_structured(provider_model_endpoint.endpoint)
    assert text
    assert structured is None or structured.location


def test_realtime_structured_thinking(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we execute the structured thinking scenario
    THEN both the travel plan and the derived budget are populated
    """

    plan, budget = run_structured_thinking(provider_model_endpoint.endpoint)
    assert plan.destination
    assert budget.items


def test_realtime_function_calling(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we invoke the function-calling scenario
    THEN at least one tool-aware choice is returned
    """

    response = run_function_calling(provider_model_endpoint.endpoint)
    assert response.choices


def test_realtime_structured_output_single_turn(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we request a single-turn structured review
    THEN the resulting payload lists product pros
    """

    review = run_structured_output_single_turn(provider_model_endpoint.endpoint)
    assert review.pros


def test_realtime_structured_output_all_providers(realtime_resource_config):
    """
    GIVEN the realtime resource configuration
    WHEN we sweep structured output across providers
    THEN every visited provider returns structured content
    """

    results = run_structured_output_all_providers(realtime_resource_config)
    assert results


def test_realtime_image_generation(provider_model_endpoint):
    """
    GIVEN a configured provider/model endpoint
    WHEN we request image generation
    THEN the provider returns a non-empty binary payload
    """

    payload = run_image_generation(provider_model_endpoint.endpoint)
    assert len(payload) > 0


def _assert_artifacts_written(tracker: ArtifactTracker):
    assert tracker.paths, "No artifact config was issued for this scenario"
    for path in tracker.paths:
        resolved = Path(path)
        assert resolved.exists(), f"Artifact root {resolved} missing"
        contents = list(resolved.rglob("*"))
        assert any(item.is_file() for item in contents), f"No files captured under {resolved}"


def _assistant_turns(messages):
    return [msg for msg in messages if getattr(msg, "role", None) == "assistant" or isinstance(msg, ChatResponse)]


class TestExampleIntents:
    def test_messages_api_flow(self, provider_model_endpoint, tmp_path):
        """
        GIVEN a provider/model endpoint
        WHEN we run the messages API intent example
        THEN user and assistant turns are produced and artifacts are persisted
        """

        tracker = ArtifactTracker(tmp_path / provider_model_endpoint.slug)
        messages = run_messages_api(provider_model_endpoint.endpoint, artifact_tracker=tracker)
        user_turns = [msg for msg in messages if isinstance(msg, Prompt) and msg.role == "user"]
        assert len(user_turns) == 2
        assistant_turns = _assistant_turns(messages)
        assert assistant_turns, "Assistant responses missing"
        _assert_artifacts_written(tracker)

    def test_streaming_messages_flow(self, provider_model_endpoint, tmp_path):
        """
        GIVEN a provider/model endpoint
        WHEN we stream the messaging example intent
        THEN both user turns and assistant turns are recorded alongside artifacts
        """

        tracker = ArtifactTracker(tmp_path / provider_model_endpoint.slug)
        messages = run_messages_streaming(provider_model_endpoint.endpoint, artifact_tracker=tracker)
        user_turns = [msg for msg in messages if isinstance(msg, Prompt) and msg.role == "user"]
        assert len(user_turns) == 2
        assistant_turns = _assistant_turns(messages)
        assert assistant_turns
        _assert_artifacts_written(tracker)

    def test_tools_with_messages_flow(self, provider_model_endpoint, tmp_path):
        """
        GIVEN a provider/model endpoint
        WHEN we execute the tools intent example
        THEN tool call results are captured and artifacts are written
        """

        tracker = ArtifactTracker(tmp_path / provider_model_endpoint.slug)
        messages = run_tools_with_messages(provider_model_endpoint.endpoint, artifact_tracker=tracker)
        tool_results = [msg for msg in messages if isinstance(msg, ToolCallResult)]
        assert tool_results, "Expected at least one tool call"
        _assert_artifacts_written(tracker)

    def test_structured_output_messages_flow(self, provider_model_endpoint, tmp_path):
        """
        GIVEN a provider/model endpoint
        WHEN we request structured analysis in the intent example
        THEN sentiment data exists and artifacts are persisted
        """

        tracker = ArtifactTracker(tmp_path / provider_model_endpoint.slug)
        structured = run_structured_output_messages(provider_model_endpoint.endpoint, artifact_tracker=tracker)
        assert structured.summary
        assert structured.sentiment in {"positive", "negative", "neutral", "mixed"}
        _assert_artifacts_written(tracker)

    def test_streaming_tools_structured_flow(self, provider_model_endpoint, tmp_path):
        """
        GIVEN a provider/model endpoint
        WHEN we stream the structured tools example
        THEN text output exists and artifacts for tool exchanges are written
        """

        tracker = ArtifactTracker(tmp_path / provider_model_endpoint.slug)
        text, structured = run_streaming_tools_structured(provider_model_endpoint.endpoint, artifact_tracker=tracker)
        assert text
        if structured:
            assert structured.location
        _assert_artifacts_written(tracker)

    def test_structured_thinking_flow(self, provider_model_endpoint, tmp_path):
        """
        GIVEN a provider/model endpoint
        WHEN we run the structured thinking intent example
        THEN both plan and budget data are produced alongside artifacts
        """

        tracker = ArtifactTracker(tmp_path / provider_model_endpoint.slug)
        plan, budget = run_structured_thinking(provider_model_endpoint.endpoint, artifact_tracker=tracker)
        assert plan.destination
        assert budget.items
        assert budget.total >= 0
        _assert_artifacts_written(tracker)
