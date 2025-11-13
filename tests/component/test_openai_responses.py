"""Component tests for OpenAIResponses request building and parsing."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from dhenara.ai.providers.base.streaming_manager import StreamingManager
from dhenara.ai.providers.openai.formatter import OpenAIFormatter
from dhenara.ai.providers.openai.message_converter import OpenAIMessageConverter
from dhenara.ai.providers.openai.responses import OpenAIResponses
from dhenara.ai.types.genai import (
    ChatMessageContentPart,
    ChatResponseReasoningContentItem,
    ChatResponseTextContentItem,
    ChatResponseUsage,
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    ToolChoice,
    ToolDefinition,
)
from dhenara.ai.types.genai.dhenara.request import Prompt, StructuredOutputConfig


def _make_responses_client(text_endpoint, call_config, *, streaming: bool = False) -> OpenAIResponses:
    """Utility to build a responses client with validation already complete."""

    config = call_config.model_copy()
    config.streaming = streaming
    client = OpenAIResponses(model_endpoint=text_endpoint, config=config, is_async=False)
    client._client = object()  # Bypass "client not initialized" guard
    client._input_validation_pending = False
    return client


@pytest.mark.component
@pytest.mark.case_id("DAI-022")
def test_build_responses_input_messages_and_prompt(monkeypatch, text_endpoint, default_call_config):
    """Ensure messages take priority over prompt/context and are converted via formatter."""

    client = _make_responses_client(text_endpoint, default_call_config)

    captured: dict[str, object] = {}

    def fake_convert(cls, message_item, model_endpoint=None, **kwargs):
        captured["message_item"] = message_item
        return {"role": "user", "content": "converted"}

    monkeypatch.setattr(OpenAIFormatter, "convert_dai_message_item_to_provider", classmethod(fake_convert))

    messages = [Prompt.with_text("hello world")]
    result = client._build_responses_input(
        prompt={"role": "user", "content": "ignored"},
        context=[{"role": "system", "content": "ignored"}],
        messages=messages,
    )

    if captured.get("message_item") is not messages[0]:
        pytest.fail("Formatter should receive the provided message item")

    if result != [{"role": "user", "content": "converted"}]:
        pytest.fail("Messages should be converted via formatter when provided")

    no_message_result = client._build_responses_input(
        prompt={"role": "user", "content": "prompt"},
        context=[{"role": "system", "content": "context"}],
        messages=None,
    )

    expected = [{"role": "system", "content": "context"}, {"role": "user", "content": "prompt"}]
    if no_message_result != expected:
        pytest.fail("When no messages are supplied, prompt and context should be preserved in order")


@pytest.mark.component
@pytest.mark.case_id("DAI-023")
def test_structured_output_uses_text_format_json_schema(text_endpoint, default_call_config):
    """Structured output should be represented via text.format JSON schema payload."""

    class SampleModel(BaseModel):
        foo: int

    config = default_call_config.model_copy()
    config.structured_output = StructuredOutputConfig.from_model(SampleModel)
    client = _make_responses_client(text_endpoint, config)

    params = client.get_api_call_params(prompt=None, context=None, instructions=None, messages=None)
    args = params.get("response_args") if isinstance(params, dict) else None

    if not isinstance(args, dict):
        pytest.fail("response_args should be returned as a dictionary")

    text_format = args.get("text")
    if not isinstance(text_format, dict):
        pytest.fail("Structured output should populate the 'text' argument")

    formatter = text_format.get("format")
    if not isinstance(formatter, dict):
        pytest.fail("text.format must be a dictionary containing schema metadata")

    if formatter.get("type") != "json_schema":
        pytest.fail("Structured output must declare a json_schema format")

    if formatter.get("name") != "SampleModel":
        pytest.fail("Schema name should match the Pydantic model title")

    if formatter.get("strict") is not True:
        pytest.fail("Structured output json_schema should enforce strict mode")

    schema = formatter.get("schema")
    if not isinstance(schema, dict) or "properties" not in schema:
        pytest.fail("Structured output schema must include properties")

    foo_schema = schema["properties"].get("foo") if isinstance(schema.get("properties"), dict) else None
    if not isinstance(foo_schema, dict) or foo_schema.get("type") != "integer":
        pytest.fail("Structured output schema should carry field type information")

    if args.get("stream") is not False:
        pytest.fail("Streaming flag should remain disabled for non-streaming config")


@pytest.mark.component
@pytest.mark.case_id("DAI-024")
def test_tools_and_tool_choice_in_args(text_endpoint, default_call_config):
    """Tools and tool_choice should be translated into Responses API payload."""

    params = FunctionParameters(
        properties={
            "path": FunctionParameter(type="string", required=True, description="Path to fetch"),
        },
        required=["path"],
    )
    function_def = FunctionDefinition(name="fetch_signal", description="Fetch verification data", parameters=params)
    tool_def = ToolDefinition(function=function_def)

    config = default_call_config.model_copy()
    config.tools = [tool_def]
    config.tool_choice = ToolChoice(type="specific", specific_tool_name="fetch_signal")

    client = _make_responses_client(text_endpoint, config)
    params = client.get_api_call_params(prompt=None, context=None, instructions=None, messages=None)
    args = params.get("response_args") if isinstance(params, dict) else None

    if not isinstance(args, dict):
        pytest.fail("response_args should be returned as a dictionary")

    tools_payload = args.get("tools")
    if not isinstance(tools_payload, list) or not tools_payload:
        pytest.fail("Tools payload should be a non-empty list")

    first_tool = tools_payload[0]
    if first_tool.get("name") != "fetch_signal":
        pytest.fail("Tool definition should expose the function name")

    parameters = first_tool.get("parameters")
    prop_type = (parameters or {}).get("properties", {}).get("path", {}).get("type") if parameters else None
    if prop_type != "string":
        pytest.fail("Tool parameters should retain type metadata for arguments")

    tool_choice = args.get("tool_choice")
    if not isinstance(tool_choice, dict) or tool_choice.get("name") != "fetch_signal":
        pytest.fail("Tool choice should target the specific function name")


@pytest.mark.component
@pytest.mark.case_id("DAI-025")
def test_parse_response_content_and_usage(monkeypatch, text_endpoint, default_call_config):
    """Parsing should wrap content items and usage metadata into ChatResponse."""

    client = OpenAIResponses(model_endpoint=text_endpoint, config=default_call_config, is_async=False)

    usage_obj = ChatResponseUsage(total_tokens=20, prompt_tokens=8, completion_tokens=12)

    def fake_usage(self, response):
        return usage_obj

    monkeypatch.setattr(OpenAIResponses, "_get_usage_from_provider_response", fake_usage, raising=False)

    text_item = ChatResponseTextContentItem(
        index=0,
        role="assistant",
        message_id="msg-1",
        message_contents=[ChatMessageContentPart(type="output_text", text="hello", annotations=None)],
    )

    def fake_convert(message, role, index_start, ai_model_provider, structured_output_config):
        return [text_item]

    monkeypatch.setattr(OpenAIMessageConverter, "provider_message_to_dai_content_items", staticmethod(fake_convert))

    class _FakeResponse:
        def __init__(self):
            self.model = "responses-1"
            self.output = ["ignored"]
            self.status = "completed"
            self.incomplete_details = None
            self.id = "resp_1"
            self.created = 123
            self.object = "response"

        def model_dump(self):
            return {"id": self.id, "object": self.object}

    fake_response = _FakeResponse()
    chat_response = client.parse_response(fake_response)

    if chat_response.usage is not usage_obj:
        pytest.fail("Usage object from parser should match helper result")

    if not chat_response.choices or chat_response.choices[0].contents != [text_item]:
        pytest.fail("Parsed choices should contain the expected content item list")

    provider_meta = getattr(chat_response.metadata, "provider_metadata", None)
    if not isinstance(provider_meta, dict) or provider_meta.get("id") != fake_response.id:
        pytest.fail("Metadata should capture provider response identifiers")

    if chat_response.provider_response != fake_response.model_dump():
        pytest.fail("Provider response should be serialized via model_dump")

    if chat_response.usage_charge is None:
        pytest.fail("Usage charge should be calculated when cost tracking is enabled")


@pytest.mark.component
@pytest.mark.case_id("DAI-026")
def test_parse_stream_chunk_text_and_reasoning_deltas(text_endpoint, default_call_config):
    """Streaming chunks should incrementally build text and reasoning content."""

    client = _make_responses_client(text_endpoint, default_call_config, streaming=True)
    client.streaming_manager = StreamingManager(model_endpoint=text_endpoint, structured_output_config=None)

    message_add = SimpleNamespace(
        type="response.output_item.added",
        item=SimpleNamespace(type="message", id="msg-1", content=[]),
        output_index=0,
        id="chunk-add",
        created=0,
        object="event",
    )
    client.parse_stream_chunk(message_add)

    text_delta = SimpleNamespace(
        type="response.output_text.delta",
        delta="Hello",
        output_index=0,
        id="chunk-text",
        created=0,
        object="event",
    )
    text_results = client.parse_stream_chunk(text_delta)

    if not text_results:
        pytest.fail("Text delta should yield streaming ChatResponse updates")

    streamed_choices = getattr(client.streaming_manager, "choices", [])
    if not streamed_choices or not streamed_choices[0].contents:
        pytest.fail("Streaming manager should accumulate choice contents after text delta")

    text_item = streamed_choices[0].contents[0]
    text_parts = getattr(text_item, "message_contents", []) or []
    text_value = text_parts[0].text if text_parts else None
    if text_value != "Hello":
        pytest.fail("Accumulated streaming text should match emitted delta")

    reasoning_add = SimpleNamespace(
        type="response.output_item.added",
        item=SimpleNamespace(type="reasoning", id="think-1", content=None, summary=None),
        output_index=0,
        id="chunk-reason-add",
        created=0,
        object="event",
    )
    client.parse_stream_chunk(reasoning_add)

    reasoning_delta = SimpleNamespace(
        type="response.reasoning_text.delta",
        delta="Thinking",
        item_id="think-1",
        content_index=0,
        output_index=0,
        id="chunk-reason",
        created=0,
        object="event",
    )
    reasoning_results = client.parse_stream_chunk(reasoning_delta)

    if not reasoning_results:
        pytest.fail("Reasoning delta should yield streaming updates")

    streamed_choices = getattr(client.streaming_manager, "choices", [])
    if not streamed_choices or not streamed_choices[0].contents:
        pytest.fail("Streaming manager should retain reasoning content after delta")

    reasoning_item = None
    for content in streamed_choices[0].contents:
        if isinstance(content, ChatResponseReasoningContentItem):
            reasoning_item = content
            break

    if reasoning_item is None:
        pytest.fail("Reasoning delta should create a reasoning content item")

    reasoning_parts = getattr(reasoning_item, "message_contents", []) or []
    reasoning_text = reasoning_parts[0].text if reasoning_parts else None
    if reasoning_text != "Thinking":
        pytest.fail("Reasoning text should accumulate deltas emitted by the provider")
