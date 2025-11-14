from __future__ import annotations

import base64
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from dhenara.ai import AIModelClient
from dhenara.ai.testing import ArtifactKind, ArtifactMetadata
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint, ChatResponse, ImageContentFormat, ResourceConfig
from dhenara.ai.types.conversation import ConversationNode
from dhenara.ai.types.genai.dhenara import PromptMessageRoleEnum
from dhenara.ai.types.genai.dhenara.request import (
    ArtifactConfig,
    Content,
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    MessageItem,
    Prompt,
    PromptText,
    SystemInstruction,
    ToolCallResult,
    ToolChoice,
    ToolDefinition,
)
from dhenara.ai.types.shared import SSEErrorResponse, SSEEventType, SSEResponse

from ._artifacts import get_artifact_manager
from ._runtime import track_scenario

_ARTIFACT_MANAGER = get_artifact_manager()
_SUITE_HINTS: dict[str, str] = {
    "packages/dhenara_ai": "dhenara_ai",
    "packages/dhenara_ai_loop": "dhenara_ai_loop",
    "verif_angels/verifinder": "verifinder",
    "verif_angels/verifinder_cli": "verifinder_cli",
    "be/fast_api": "fastapi",
}


def _sanitize_component(value: str | None, *, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    sanitized = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip()]
    result = "".join(sanitized).strip("_")
    return result or fallback


def _suite_segment() -> str | None:
    explicit = os.environ.get("DAI_TEST_ARTIFACT_SUITE")
    if explicit:
        return _sanitize_component(explicit)

    node_id = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
    if "::" in node_id:
        node_id = node_id.split("::", 1)[0]
    normalized = node_id.replace("\\", "/")
    for prefix, slug in _SUITE_HINTS.items():
        if normalized.startswith(prefix):
            return slug
    cwd_name = Path.cwd().name
    return _sanitize_component(cwd_name) if cwd_name else None


def _current_test_segments() -> list[str]:
    """Return a simple `<module>/<test_fn>` style path for artifacts.

    Examples
    --------
    - test_all_capabilities.py::test_realtime_text_generation[open_ai-gpt-5-nano]
      -> ["test_all_capabilities", "test_realtime_text_generation"]

    - test_all_providers.py::test_realtime_structured_output_all_providers
      -> ["test_all_providers", "test_realtime_structured_output_all_providers"]
    """

    node_id = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
    if not node_id:
        return []

    parts = node_id.split("::")
    if not parts:
        return []

    # Normalised module name (strip .py and suffixes like _py)
    module_path = Path(parts[0])
    module_stem = module_path.stem  # e.g. "test_all_capabilities"
    module_segment = _sanitize_component(module_stem)

    # Base test function name (drop any param suffix [..])
    test_name: str | None = None
    if len(parts) > 1:
        func_part = parts[1]
        if "[" in func_part:
            func_part = func_part.split("[", 1)[0]
        test_name = _sanitize_component(func_part)

    segments: list[str] = []
    if module_segment:
        segments.append(module_segment)
    if test_name:
        segments.append(test_name)
    return segments


def _provider_segment(endpoint: AIModelEndpoint | None = None, *, slug: str | None = None) -> str | None:
    if slug:
        return _sanitize_component(slug)
    if not endpoint:
        return None
    api = getattr(endpoint, "api", None)
    provider = getattr(api, "provider", None)
    provider_value = getattr(provider, "value", str(provider)) if provider else None
    model = getattr(endpoint, "ai_model", None)
    model_name = getattr(model, "model_name", None)
    provider_slug = _sanitize_component(str(provider_value).replace(".", "-")) if provider_value else None
    model_slug = _sanitize_component(model_name.replace(".", "-")) if model_name else None
    if provider_slug and model_slug:
        return f"{provider_slug}__{model_slug}"
    return provider_slug or model_slug


class ArtifactTracker:
    def __init__(self, base_dir: Path | str | None = None, *, default_kind: ArtifactKind = ArtifactKind.LOG):
        self._base = Path(base_dir) if base_dir else None
        self.paths: list[Path] = []
        self._default_kind = default_kind
        self._manager = _ARTIFACT_MANAGER

    def configure(
        self,
        scenario: str,
        label: str | None = None,
        *,
        kind: ArtifactKind | None = None,
        provider: str | None = None,
    ) -> ArtifactConfig | None:
        base_root = self._base or getattr(self._manager, "base_dir", None)
        if not base_root:
            return None
        base_root = Path(base_root)
        if not self._base:
            suite_dir = _suite_segment()
            if suite_dir:
                base_root = base_root / suite_dir
            if provider:
                base_root = base_root / provider
            # Use simple `<module>/<test_fn>` structure for all realtime tests
            for segment in _current_test_segments():
                base_root = base_root / segment
        relative = Path(scenario)
        if label:
            relative = relative / label
        artifact_root = Path(base_root) / relative
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.paths.append(artifact_root)
        metadata = ArtifactMetadata(path=artifact_root, kind=kind or self._default_kind, label=label or scenario)
        run_id = f"{scenario}:{label}" if label else scenario
        self._manager.register(run_id=run_id, metadata=metadata)
        self._manager.flush()
        return ArtifactConfig(
            enabled=True,
            artifact_root=str(artifact_root),
            capture_dhenara_request=True,
            capture_provider_request=True,
            capture_provider_response=True,
            capture_dhenara_response=True,
            enable_python_logs=True,
        )


_DEFAULT_ARTIFACT_TRACKER = ArtifactTracker()


def _artifact_config_for(
    scenario: str,
    label: str | None = None,
    *,
    artifact_tracker: ArtifactTracker | None = None,
    kind: ArtifactKind = ArtifactKind.LOG,
    endpoint: AIModelEndpoint | None = None,
    provider_slug: str | None = None,
) -> ArtifactConfig | None:
    tracker = artifact_tracker or _DEFAULT_ARTIFACT_TRACKER
    provider = provider_slug or _provider_segment(endpoint)
    return tracker.configure(scenario, label, kind=kind, provider=provider)


@dataclass
class StreamResult:
    final: ChatResponse
    deltas: list[str]


def _status_reason(response, fallback: str) -> str:
    status = getattr(response, "status", None)
    if not status:
        return fallback
    provider = getattr(status, "api_provider", "provider")
    model = getattr(status, "model", "model")
    message = getattr(status, "message", fallback)
    code = getattr(status, "code", None) or getattr(status, "http_status_code", None)
    suffix = f" code={code}" if code else ""
    return f"{fallback} ({provider}/{model}:{suffix} {message})"


def _skip_scenario(scenario: str, message: str, response=None) -> None:
    reason = f"{scenario} skipped: {message}"
    if response is not None:
        try:
            reason = _status_reason(response, reason)
        except Exception:  # pragma: no cover - best effort context
            pass
    pytest.skip(reason)


def _require_chat_response(response, scenario: str) -> ChatResponse:
    if response.chat_response:
        return response.chat_response
    _skip_scenario(scenario, "provider returned no chat response", response)


def _require_structured(
    data: dict[str, Any] | None,
    scenario: str,
    response: ChatResponse | None = None,
) -> dict[str, Any]:
    if data:
        return data
    _skip_scenario(scenario, "provider omitted structured output", response)


def _consume_stream(response, scenario: str) -> StreamResult:
    if not response.stream_generator:
        raise AssertionError("Streaming generator missing on response")

    collected: list[str] = []
    final_response: ChatResponse | None = None
    last_response: ChatResponse | None = None

    for chunk, final in response.stream_generator:
        if isinstance(chunk, SSEErrorResponse):
            _skip_scenario(scenario, f"streaming error: {chunk.data.message}", last_response)
        if isinstance(chunk, SSEResponse) and chunk.event == SSEEventType.ERROR:
            _skip_scenario(scenario, "streaming error event", last_response)
        if hasattr(chunk, "choice_deltas"):
            for choice_delta in chunk.choice_deltas:
                if not choice_delta.content_deltas:
                    continue
                for content_delta in choice_delta.content_deltas:
                    text_delta = getattr(content_delta, "get_text_delta", None)
                    if callable(text_delta):
                        delta_text = text_delta()
                        if delta_text:
                            collected.append(delta_text)
        if final and final.chat_response:
            final_response = final.chat_response
            last_response = final.chat_response

    if not final_response:
        _skip_scenario(scenario, "streaming unavailable: missing final response")

    return StreamResult(final=final_response, deltas=collected)


async def _consume_async_stream(response, scenario: str) -> StreamResult:
    if not response.stream_generator:
        raise AssertionError("Streaming generator missing on response")

    collected: list[str] = []
    final_response: ChatResponse | None = None
    last_response: ChatResponse | None = None

    async for item in response.stream_generator:
        if isinstance(item, tuple):
            chunk, final = item
        else:  # pragma: no cover - defensive guard
            chunk, final = item, None

        if isinstance(chunk, SSEErrorResponse):
            _skip_scenario(scenario, f"streaming error: {chunk.data.message}", last_response)
        if isinstance(chunk, SSEResponse) and chunk.event == SSEEventType.ERROR:
            _skip_scenario(scenario, "streaming error event", last_response)

        if hasattr(chunk, "choice_deltas"):
            for choice_delta in chunk.choice_deltas:
                if not choice_delta.content_deltas:
                    continue
                for content_delta in choice_delta.content_deltas:
                    text_delta = getattr(content_delta, "get_text_delta", None)
                    if callable(text_delta):
                        delta_text = text_delta()
                        if delta_text:
                            collected.append(delta_text)

        if final and getattr(final, "chat_response", None):
            final_response = final.chat_response
            last_response = final.chat_response

    if not final_response:
        _skip_scenario(scenario, "streaming unavailable: missing final response")

    return StreamResult(final=final_response, deltas=collected)


def _ensure_text(response: ChatResponse, scenario: str = "text_output") -> str:
    text = response.text()
    if not text or not text.strip():
        _skip_scenario(scenario, "text output missing from provider response", response)
    return text.strip()


def _build_context(history: Iterable[ConversationNode]) -> list[Prompt]:
    context: list[Prompt] = []
    for node in history:
        context.extend(node.get_context())
    return context


def _append_node(history: list[ConversationNode], user_query: str, response: ChatResponse) -> ConversationNode:
    node = ConversationNode(user_query=user_query, input_files=[], response=response)
    history.append(node)
    return node


def _build_client(
    endpoint: AIModelEndpoint,
    *,
    streaming: bool = False,
    reasoning: bool = False,
    structured_output: type[BaseModel] | None = None,
    tools: Sequence[ToolDefinition] | None = None,
    tool_choice: ToolChoice | dict[str, Any] | None = None,
    is_async: bool = False,
    options: dict[str, Any] | None = None,
    max_output_tokens: int = 2000,
    max_reasoning_tokens: int = 1024,
    artifact_config: ArtifactConfig | None = None,
) -> AIModelClient:
    config = AIModelCallConfig(
        streaming=streaming,
        max_output_tokens=None if options else max_output_tokens,
        reasoning=reasoning,
        max_reasoning_tokens=max_reasoning_tokens if reasoning else None,
        reasoning_effort="low" if reasoning else None,
        structured_output=structured_output,
        tools=list(tools) if tools else None,
        tool_choice=tool_choice,
        options=options or {},
        artifact_config=artifact_config,
    )
    return AIModelClient(model_endpoint=endpoint, config=config, is_async=is_async)


# ---------------------------------------------------------------------------
# Tooling shared across scenarios
# ---------------------------------------------------------------------------


def _tool_get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    temperature_c = 22 if unit == "celsius" else 72
    return {
        "location": location,
        "temperature": temperature_c,
        "unit": unit,
        "condition": "sunny",
        "humidity": 60,
    }


def _tool_calculate(operation: str, a: float, b: float) -> dict[str, Any]:
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b else None
    else:
        result = None
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result,
    }


_WEATHER_TOOL = ToolDefinition(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters=FunctionParameters(
            type="object",
            required=["location"],
            properties={
                "location": FunctionParameter(type="string", description="City and state"),
                "unit": FunctionParameter(
                    type="string",
                    description="Temperature unit",
                    allowed_values=["celsius", "fahrenheit"],
                ),
            },
        ),
    )
)

_CALCULATOR_TOOL = ToolDefinition(
    function=FunctionDefinition(
        name="calculate",
        description="Perform basic arithmetic operations",
        parameters=FunctionParameters(
            type="object",
            required=["operation", "a", "b"],
            properties={
                "operation": FunctionParameter(
                    type="string",
                    description="Operation",
                    allowed_values=["add", "subtract", "multiply", "divide"],
                ),
                "a": FunctionParameter(type="number", description="First operand"),
                "b": FunctionParameter(type="number", description="Second operand"),
            },
        ),
    )
)

_TOOL_REGISTRY = {
    "get_weather": _tool_get_weather,
    "calculate": _tool_calculate,
}


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------


class WeatherInfo(BaseModel):
    location: str
    temperature: float
    condition: str
    wind_speed: float


class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str
    hobbies: list[str]


class StoryAnalysis(BaseModel):
    title: str
    main_characters: list[str]
    themes: list[str]
    sentiment: str
    summary: str


class ProductRatings(BaseModel):
    rating: int = Field(ge=1, le=5)
    value_for_money_rating: int = Field(ge=1, le=5)


class ProductReview(BaseModel):
    product_name: str
    rating: ProductRatings
    pros: list[str]
    cons: list[str]
    summary: str


class TravelPlan(BaseModel):
    destination: str
    days: int = Field(ge=1, le=14)
    interests: list[str] = Field(min_length=1)


class BudgetItem(BaseModel):
    name: str
    cost: float = Field(ge=0)


class Budget(BaseModel):
    currency: str
    items: list[BudgetItem]
    total: float


class CompactWeatherCard(BaseModel):
    location: str
    temperature_c: float
    condition: str


# ---------------------------------------------------------------------------
# Core scenario runners
# ---------------------------------------------------------------------------


def run_text_generation_sync(endpoint: AIModelEndpoint) -> ChatResponse:
    with track_scenario(endpoint, "text_generation"):
        artifact_config = _artifact_config_for("text_generation", endpoint=endpoint)
        client = _build_client(
            endpoint,
            streaming=False,
            reasoning=True,
            artifact_config=artifact_config,
        )
        response = client.generate(
            prompt="List two productivity practices that help a solo engineer stay focused.",
            instructions=["Keep the answer under 80 words."],
        )
        chat_response = _require_chat_response(response, "text_generation")
        _ensure_text(chat_response, "text_generation")
        return chat_response


def run_text_streaming_sync(endpoint: AIModelEndpoint) -> StreamResult:
    with track_scenario(endpoint, "text_streaming"):
        artifact_config = _artifact_config_for("text_streaming", endpoint=endpoint)
        client = _build_client(
            endpoint,
            streaming=True,
            reasoning=True,
            artifact_config=artifact_config,
        )
        response = client.generate(
            prompt="Describe sunrise over a futuristic city in two sentences.",
            instructions=["Make it vivid but concise."],
        )
        stream = _consume_stream(response, "text_streaming")
        _ensure_text(stream.final, "text_streaming")
        if not stream.deltas:
            pytest.fail("streaming unavailable: provider returned no token deltas")
        return stream


def run_various_input_formats(endpoint: AIModelEndpoint) -> list[str]:
    with track_scenario(endpoint, "input_formats"):
        artifact_config = _artifact_config_for("input_formats", endpoint=endpoint)
        client = _build_client(
            endpoint,
            streaming=False,
            reasoning=False,
            artifact_config=artifact_config,
        )
        outputs: list[str] = []

        user_prompt = "Share three debugging tactics for complex async Python code."

        resp1 = client.generate(prompt=user_prompt, instructions=["Respond in <=60 words."])
        outputs.append(_ensure_text(_require_chat_response(resp1, "input_formats/plain"), "input_formats/plain"))

        resp2 = client.generate(
            prompt={"role": "user", "text": user_prompt},
            context=[],
            instructions=["Respond in <=60 words."],
        )
        outputs.append(_ensure_text(_require_chat_response(resp2, "input_formats/legacy"), "input_formats/legacy"))

        content = Content(type="text", text=user_prompt)
        prompt_obj = Prompt(role=PromptMessageRoleEnum.USER, text=PromptText(content=content))
        resp3 = client.generate(prompt=prompt_obj, instructions=[SystemInstruction(text="Respond in <=60 words.")])
        resp3_chat = _require_chat_response(resp3, "input_formats/object")
        outputs.append(_ensure_text(resp3_chat, "input_formats/object"))

        context_message = resp3_chat.to_prompt()
        resp4 = client.generate(
            prompt="Drill into the first tactic with one actionable step.",
            context=[context_message] if context_message else [],
            instructions=["Keep it under 30 words."],
        )
        outputs.append(_ensure_text(_require_chat_response(resp4, "input_formats/context"), "input_formats/context"))

        return outputs


def run_multi_turn_conversation(endpoint: AIModelEndpoint) -> list[ConversationNode]:
    with track_scenario(endpoint, "multi_turn"):
        history: list[ConversationNode] = []
        artifact_config = _artifact_config_for("multi_turn", "turns", endpoint=endpoint)
        client = _build_client(
            endpoint,
            streaming=False,
            reasoning=True,
            artifact_config=artifact_config,
        )

        turns = [
            ("Help me name a verification dashboard for pre-silicon regressions.", ["Offer 3 succinct options."]),
            ("Pick the catchiest option and craft a two-sentence pitch.", ["Keep the tone energetic."]),
            ("Summarise why this dashboard matters for ASIC teams.", ["Highlight automation benefits."]),
        ]

        for prompt_text, instructions in turns:
            context = _build_context(history)
            response = client.generate(prompt=prompt_text, context=context, instructions=instructions)
            chat_response = _require_chat_response(response, "multi_turn")
            _append_node(history, prompt_text, chat_response)

        return history


def run_streaming_multi_turn(endpoint: AIModelEndpoint) -> list[ConversationNode]:
    with track_scenario(endpoint, "streaming_multi_turn"):
        history: list[ConversationNode] = []
        artifact_config = _artifact_config_for("streaming_multi_turn", "turns", endpoint=endpoint)
        client = _build_client(
            endpoint,
            streaming=True,
            reasoning=True,
            artifact_config=artifact_config,
        )

        queries = [
            "Outline a short customer story for verification tooling.",
            "Add a twist that involves automated log triage.",
        ]

        for query in queries:
            context = _build_context(history)
            response = client.generate(prompt=query, context=context, instructions=["Limit to 120 words."])
            result = _consume_stream(response, "streaming_multi_turn")
            history.append(ConversationNode(user_query=query, input_files=[], response=result.final))

        return history


async def run_async_streaming_multi_turn(endpoint: AIModelEndpoint) -> list[ConversationNode]:
    with track_scenario(endpoint, "async_streaming_multi_turn"):
        history: list[ConversationNode] = []
        artifact_config = _artifact_config_for("async_streaming_multi_turn", "turns", endpoint=endpoint)
        client = _build_client(
            endpoint,
            streaming=True,
            reasoning=True,
            is_async=True,
            artifact_config=artifact_config,
        )

        queries = [
            "Draft a 30-word release note about improved regression clustering.",
            "Compose a matching follow-up tweet in 25 words or fewer.",
        ]

        for query in queries:
            context = _build_context(history)
            response = await client.generate_async(
                prompt=query,
                context=context,
                instructions=["Use plain language."],
            )
            result = await _consume_async_stream(response, "async_streaming_multi_turn")
            history.append(ConversationNode(user_query=query, input_files=[], response=result.final))

        await client.cleanup_async()
        return history


def run_messages_api(
    endpoint: AIModelEndpoint,
    *,
    artifact_tracker: ArtifactTracker | None = None,
) -> list[MessageItem]:
    with track_scenario(endpoint, "messages_api"):
        messages: list[MessageItem] = []
        artifact_config = _artifact_config_for(
            "example14_messages",
            "turns",
            artifact_tracker=artifact_tracker,
            endpoint=endpoint,
        )
        client = _build_client(
            endpoint,
            streaming=False,
            reasoning=True,
            artifact_config=artifact_config,
        )

        prompts = [
            ("Tell me a short story about an agent debugging RTL regressions.", ["Limit to 80 words."]),
            ("Add a surprising twist involving a flaky test.", ["Keep continuity."]),
        ]

        for prompt_text, instructions in prompts:
            run_messages = [*messages, Prompt(role="user", text=prompt_text)]
            response = client.generate(messages=run_messages, instructions=instructions)
            chat_response = _require_chat_response(response, "messages_api")
            assistant_item = chat_response.to_message_item()
            if assistant_item:
                messages.append(assistant_item)
            messages.append(Prompt(role="user", text=prompt_text))

        return messages


def run_messages_streaming(
    endpoint: AIModelEndpoint,
    *,
    artifact_tracker: ArtifactTracker | None = None,
) -> list[MessageItem]:
    with track_scenario(endpoint, "messages_streaming"):
        messages: list[MessageItem] = []
        artifact_config = _artifact_config_for(
            "example15_streaming",
            "turns",
            artifact_tracker=artifact_tracker,
            endpoint=endpoint,
        )
        client = _build_client(
            endpoint,
            streaming=True,
            reasoning=True,
            artifact_config=artifact_config,
        )

        queries = [
            "Pitch verifinder to a verification lead in two sentences.",
            "Respond to a follow-up asking about log summarisation support.",
        ]

        for query in queries:
            payload = [*messages, Prompt(role="user", text=query)]
            response = client.generate(messages=payload, instructions=["Stay energetic."])
            result = _consume_stream(response, "messages_streaming")
            assistant_item = result.final.to_message_item()
            if assistant_item:
                messages.append(assistant_item)
            messages.append(Prompt(role="user", text=query))

        return messages


def _execute_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in _TOOL_REGISTRY:
        return {"error": f"unknown tool {name}"}
    try:
        return _TOOL_REGISTRY[name](**arguments)
    except Exception as exc:  # pragma: no cover - defensive guard
        return {"error": str(exc)}


def run_tools_with_messages(
    endpoint: AIModelEndpoint,
    *,
    artifact_tracker: ArtifactTracker | None = None,
) -> list[MessageItem]:
    with track_scenario(endpoint, "tools_with_messages"):
        messages: list[MessageItem] = []
        tools = [_WEATHER_TOOL, _CALCULATOR_TOOL]
        artifact_config = _artifact_config_for(
            "example16_tools",
            "dialog",
            artifact_tracker=artifact_tracker,
            endpoint=endpoint,
        )
        client = _build_client(
            endpoint,
            streaming=False,
            reasoning=True,
            tools=tools,
            tool_choice={"type": "zero_or_more"},
            artifact_config=artifact_config,
        )

        prompt = "Should I take an umbrella for tomorrow's meetings in Austin?"
        messages.append(Prompt(role="user", text=prompt))

        for _ in range(3):
            response = client.generate(messages=messages, instructions=["Use tools when helpful."])
            chat_response = _require_chat_response(response, "tools_with_messages")
            if not chat_response.choices:
                break

            assistant_item = chat_response.to_message_item()
            if assistant_item:
                messages.append(assistant_item)

            tool_calls = chat_response.tools()
            if not tool_calls:
                break

            for item in tool_calls:
                call = getattr(item, "tool_call", None)
                if not call:
                    continue
                result_payload = _execute_tool_call(call.name, call.arguments or {})
                messages.append(
                    ToolCallResult(
                        call_id=call.call_id or call.id or "tool_call_0",
                        name=call.name,
                        output=result_payload,
                    )
                )

        return messages


def run_structured_output_messages(
    endpoint: AIModelEndpoint,
    *,
    artifact_tracker: ArtifactTracker | None = None,
) -> StoryAnalysis:
    with track_scenario(endpoint, "structured_output_messages"):
        artifact_config = _artifact_config_for(
            "example17_structured",
            "analysis",
            artifact_tracker=artifact_tracker,
            endpoint=endpoint,
        )
        client = _build_client(
            endpoint,
            streaming=False,
            reasoning=True,
            structured_output=StoryAnalysis,
            artifact_config=artifact_config,
        )

        messages: list[MessageItem] = [
            Prompt(role="user", text="Analyse a short tale about an AI assisting chip verification."),
        ]
        response = client.generate(messages=messages, instructions=["Return only structured data."])
        chat_response = _require_chat_response(response, "structured_output_messages")
        structured = _require_structured(chat_response.structured(), "structured_output_messages", chat_response)
        return StoryAnalysis(**structured)


def run_streaming_tools_structured(
    endpoint: AIModelEndpoint,
    *,
    artifact_tracker: ArtifactTracker | None = None,
) -> tuple[str, CompactWeatherCard]:
    with track_scenario(endpoint, "streaming_tools_structured"):
        messages: list[MessageItem] = [
            Prompt(role="user", text="What is the weather in Berlin and how should I prepare?")
        ]
        tools = [_WEATHER_TOOL, _CALCULATOR_TOOL]
        artifact_config = _artifact_config_for(
            "example18_streaming_tools",
            "dialog",
            artifact_tracker=artifact_tracker,
            endpoint=endpoint,
        )

        client = _build_client(
            endpoint,
            streaming=True,
            reasoning=True,
            tools=tools,
            tool_choice={"type": "zero_or_more"},
            artifact_config=artifact_config,
        )

        stream = client.generate(messages=messages, instructions=["Use tools before final answer if necessary."])
        result = _consume_stream(stream, "streaming_tools_structured")

        assistant_item = result.final.to_message_item()
        if assistant_item:
            messages.append(assistant_item)

        for tool_item in result.final.tools():
            tool_call = getattr(tool_item, "tool_call", None)
            if not tool_call:
                continue
            tool_result = _execute_tool_call(tool_call.name, tool_call.arguments or {})
            messages.append(
                ToolCallResult(
                    call_id=tool_call.call_id or tool_call.id or "tool_call_0",
                    name=tool_call.name,
                    output=tool_result,
                )
            )

        structured_data = result.final.structured()
        structured_card = None
        if structured_data:
            try:
                structured_card = CompactWeatherCard(**structured_data)
            except Exception:  # pragma: no cover - depend on model behaviour
                structured_card = None

        return (_ensure_text(result.final, "streaming_tools_structured/final"), structured_card)


def run_structured_thinking(
    endpoint: AIModelEndpoint,
    *,
    artifact_tracker: ArtifactTracker | None = None,
) -> tuple[TravelPlan, Budget]:
    with track_scenario(endpoint, "structured_thinking"):
        plan_artifact = _artifact_config_for(
            "example19_structured_thinking",
            "plan",
            artifact_tracker=artifact_tracker,
            endpoint=endpoint,
        )
        client = _build_client(
            endpoint,
            streaming=True,
            reasoning=True,
            structured_output=TravelPlan,
            artifact_config=plan_artifact,
        )

        travel_stream = client.generate(
            messages=[
                Prompt(
                    role="user",
                    text="Create a TravelPlan JSON for a 2-day trip to Tokyo focused on food tours.",
                )
            ]
        )
        travel_result = _consume_stream(travel_stream, "structured_thinking_plan")
        plan_data = _require_structured(
            travel_result.final.structured(),
            "structured_thinking/plan",
            travel_result.final,
        )
        plan = TravelPlan(**plan_data)

        budget_artifact = _artifact_config_for(
            "example19_structured_thinking",
            "budget",
            artifact_tracker=artifact_tracker,
            endpoint=endpoint,
        )
        budget_client = _build_client(
            endpoint,
            streaming=True,
            reasoning=True,
            structured_output=Budget,
            artifact_config=budget_artifact,
        )

        budget_prompt = (
            f"Using the plan for {plan.destination} lasting {plan.days} days, create a budget with at least"
            " three items and a total that sums the items."
        )
        budget_stream = budget_client.generate(messages=[Prompt(role="user", text=budget_prompt)])
        budget_result = _consume_stream(budget_stream, "structured_thinking_budget")
        budget_data = _require_structured(
            budget_result.final.structured(),
            "structured_thinking/budget",
            budget_result.final,
        )
        budget = Budget(**budget_data)

        return plan, budget


def run_function_calling(endpoint: AIModelEndpoint) -> ChatResponse:
    with track_scenario(endpoint, "function_calling"):
        artifact_config = _artifact_config_for("function_calling", endpoint=endpoint)
        schedule_tool = ToolDefinition.from_callable(
            lambda title, start_time, duration_minutes=30: {
                "title": title,
                "start_time": start_time,
                "duration_minutes": duration_minutes,
                "status": "scheduled",
            }
        )

        tools = [_WEATHER_TOOL, schedule_tool]
        client = _build_client(
            endpoint,
            streaming=False,
            reasoning=False,
            tools=tools,
            tool_choice=ToolChoice(type="one_or_more"),
            artifact_config=artifact_config,
        )

        response = client.generate(
            prompt="Should I move my design review since storms are forecast in Austin tomorrow?",
            context=[],
            instructions=["Call tools if you need external data."],
        )
        return _require_chat_response(response, "function_calling")


def run_structured_output_single_turn(endpoint: AIModelEndpoint) -> ProductReview:
    with track_scenario(endpoint, "structured_output_single"):
        artifact_config = _artifact_config_for("structured_output_single", endpoint=endpoint)
        client = _build_client(
            endpoint,
            streaming=False,
            reasoning=True,
            structured_output=ProductReview,
            artifact_config=artifact_config,
        )
        response = client.generate(
            prompt="Review the latest verifinder agent release and provide structured pros/cons.",
            context=[],
            instructions=["Only respond with structured data."],
        )
        chat_response = _require_chat_response(response, "structured_output_single")
        structured = _require_structured(chat_response.structured(), "structured_output_single", chat_response)
        return ProductReview(**structured)


def run_image_generation(endpoint: AIModelEndpoint, *, size: str = "512x512") -> bytes:
    with track_scenario(endpoint, "image_generation"):
        artifact_config = _artifact_config_for("image_generation", kind=ArtifactKind.MEDIA, endpoint=endpoint)
        client = _build_client(
            endpoint,
            options={
                "n": 1,
                "size": size,
                "response_format": "b64_json",
            },
            artifact_config=artifact_config,
        )
        response = client.generate(
            prompt="Generate an illustration of a robotics engineer reviewing simulation logs on multiple monitors.",
            context=[],
            instructions=[],
        )
        if not response.image_response:
            raise AssertionError("Image response missing")

        for choice in response.image_response.choices:
            for image_content in choice.contents:
                if image_content.content_format == ImageContentFormat.BASE64 and image_content.content_b64_json:
                    return base64.b64decode(image_content.content_b64_json)
                if image_content.content_format == ImageContentFormat.BYTES and image_content.content_bytes:
                    return image_content.content_bytes
        raise AssertionError("No usable image payload returned")


def run_structured_output_all_providers(resource_config: ResourceConfig) -> dict[str, bool]:
    results: dict[str, bool] = {}
    seen: set[str] = set()

    for endpoint in resource_config.model_endpoints:
        provider_value = getattr(endpoint.api.provider, "value", str(endpoint.api.provider))
        if provider_value in seen:
            continue

        scenario_name = f"structured_output_sweep::{provider_value}"
        with track_scenario(endpoint, scenario_name):
            artifact_config = _artifact_config_for(scenario_name, endpoint=endpoint)
            client = _build_client(
                endpoint,
                streaming=True,
                reasoning=True,
                structured_output=TravelPlan,
                artifact_config=artifact_config,
            )

            stream = client.generate(
                messages=[
                    Prompt(
                        role="user",
                        text="Create a TravelPlan JSON for a 3-day visit to Tokyo focused on ramen and culture.",
                    )
                ]
            )

            final = _consume_stream(stream, scenario_name)
            structured = final.final.structured()
            if not structured:
                pytest.fail(f"{provider_value} structured output sweep produced no data")
            results[provider_value] = True
            seen.add(provider_value)

    if not results:
        pytest.fail("No providers available for structured output sweep")

    return results
