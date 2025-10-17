"""Streaming multi-turn conversation with tools and structured output.

This example streams tool calls (with incremental args) and text output.
It also demonstrates structured output streaming by asking for a JSON schema
in a subsequent turn.

Usage:
  python examples/18_streaming_multi_turn_with_tools_and_structured_output.py
"""

import json
import random
from typing import Literal

from include.console_renderer import StreamingRenderer, render_usage
from include.shared_config import all_endpoints, create_artifact_config, generate_run_dirname, load_resource_config
from pydantic import BaseModel

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.genai.dhenara.request import (
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    MessageItem,
    Prompt,
    ToolCallResult,
    ToolDefinition,
)

# --- Mock tools ---


def get_weather(location: str, unit: str = "celsius") -> dict:
    return {"location": location, "temperature": 22, "unit": unit, "condition": "sunny", "humidity": 65}


def calculate(operation: str, a: float, b: float) -> dict:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b != 0 else None}
    return {"operation": operation, "a": a, "b": b, "result": ops.get(operation)}


weather_tool = ToolDefinition(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters=FunctionParameters(
            type="object",
            required=["location"],
            properties={
                "location": FunctionParameter(type="string", description="City and state, e.g. San Francisco, CA"),
                "unit": FunctionParameter(
                    type="string", description="Temperature unit", allowed_values=["celsius", "fahrenheit"]
                ),
            },
        ),
    )
)

calculator_tool = ToolDefinition(
    function=FunctionDefinition(
        name="calculate",
        description="Perform basic arithmetic operations",
        parameters=FunctionParameters(
            type="object",
            required=["operation", "a", "b"],
            properties={
                "operation": FunctionParameter(
                    type="string",
                    description="The operation to perform",
                    allowed_values=["add", "subtract", "multiply", "divide"],
                ),
                "a": FunctionParameter(type="number", description="First number"),
                "b": FunctionParameter(type="number", description="Second number"),
            },
        ),
    )
)

TOOL_REGISTRY = {"get_weather": get_weather, "calculate": calculate}


# Structured output schema for the second half
class WeatherInfo(BaseModel):
    location: str
    temperature: float
    condition: Literal["sunny", "cloudy", "rainy", "snowy"]
    humidity: int


resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


def execute_tool_call(tool_name: str, arguments: dict) -> dict:
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Tool {tool_name} not found"}
    try:
        return TOOL_REGISTRY[tool_name](**arguments)
    except Exception as e:
        return {"error": str(e)}


def handle_streaming_turn_with_tools(
    user_query: str,
    endpoint: AIModelEndpoint,
    messages: list[MessageItem],
    art_dir: str,
) -> tuple[str, list[MessageItem]]:
    """Single turn that streams, executes tools if emitted, and returns final text."""

    artifact_config = create_artifact_config(art_dir)

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=True,
            artifact_config=artifact_config,
            tools=[weather_tool, calculator_tool],
            tool_choice={"type": "zero_or_more"},
        ),
        is_async=False,
    )

    streaming_renderer = StreamingRenderer()

    # Build messages and call
    new_messages = [*messages, Prompt(role="user", text=user_query)]
    stream = client.generate(
        messages=new_messages,
        instructions=["Use tools when helpful; otherwise answer succinctly."],
    )

    final = streaming_renderer.process_stream(stream)

    # If no final response, return empty text and unchanged messages
    if not final or not final.choices:
        return "", messages

    # Add assistant message to history (keeps tool calls/text together)
    assistant_msg = final.to_message_item()
    if assistant_msg:
        new_messages.append(assistant_msg)

    # Execute tool calls if present and append tool results
    for choice in final.choices:
        for content in choice.contents:
            if hasattr(content, "tool_call") and content.tool_call:
                tc = content.tool_call
                res = execute_tool_call(tc.name, tc.arguments or {})
                # Prefer provider-supplied call_id; fall back to provider item id; else a deterministic placeholder
                call_id = tc.call_id or tc.id or f"tool_call_{choice.index}_{content.index}"
                tool_result = ToolCallResult(call_id=call_id, name=tc.name, output=res)
                new_messages.append(tool_result)

    # Extract first text answer for display
    text_answer = final.text() or ""
    render_usage(final)

    return text_answer, new_messages


def run_streaming_multi_turn_with_tools_and_structured_output():
    print("=" * 80)
    print("Streaming Multi-Turn with Tools + Structured Output")
    print("=" * 80)

    messages: list[MessageItem] = []
    run_dir = generate_run_dirname()

    # Turn 1: Ask weather (tool likely)
    endpoint = random.choice(resource_config.model_endpoints)
    print(f"\n🔄 Turn 1 with {endpoint.ai_model.model_name} ({endpoint.api.provider})\n")
    t1, messages = handle_streaming_turn_with_tools(
        user_query="What's the weather in Paris?",
        endpoint=endpoint,
        messages=messages,
        art_dir=f"18_stream/{run_dir}/turn_1",
    )
    print(f"Assistant: {t1}\n")

    # Turn 2: Simple math tool call
    endpoint = random.choice(resource_config.model_endpoints)
    print(f"\n🔄 Turn 2 with {endpoint.ai_model.model_name} ({endpoint.api.provider})\n")
    t2, messages = handle_streaming_turn_with_tools(
        user_query="Calculate 17 * 23.", endpoint=endpoint, messages=messages, art_dir=f"18_stream/{run_dir}/turn_2"
    )
    print(f"Assistant: {t2}\n")

    # Turn 3: Stream a structured JSON weather card
    endpoint = random.choice(resource_config.model_endpoints)
    print(f"\n🔄 Turn 3 (structured) with {endpoint.ai_model.model_name} ({endpoint.api.provider})\n")

    artifact_config = create_artifact_config(f"18_stream/{run_dir}/turn_3_struct")
    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=True,
            artifact_config=artifact_config,
            structured_output=WeatherInfo,
        ),
        is_async=False,
    )

    prompt = Prompt(
        role="user",
        text=("Produce a valid WeatherInfo JSON card for Berlin. Only output JSON. Keep values realistic."),
    )

    stream = client.generate(messages=[*messages, prompt])
    streaming_renderer = StreamingRenderer()
    final = streaming_renderer.process_stream(stream)

    if final:
        print("\nStructured output usage:")
        render_usage(final)
        data = final.structured() or {}
        try:
            validated = WeatherInfo(**data)
            print("\nStructured object:\n" + json.dumps(validated.model_dump(), indent=2))
        except Exception:
            print("\nRaw structured data:\n" + json.dumps(data, indent=2))


if __name__ == "__main__":
    run_streaming_multi_turn_with_tools_and_structured_output()
