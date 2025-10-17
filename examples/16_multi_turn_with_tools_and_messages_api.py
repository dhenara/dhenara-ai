"""Multi-turn conversation with tool calls using messages API.

This example demonstrates how to use the messages API with function calling,
showing proper handling of tool calls and tool results in the conversation flow.

Usage:
  python examples/16_multi_turn_with_tools_and_messages_api.py
"""

import json
import random

from include.shared_config import all_endpoints, create_artifact_config, generate_run_dirname, load_resource_config

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


# Mock tool functions
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get weather for a location."""
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "sunny",
        "humidity": 65,
    }


def calculate(operation: str, a: float, b: float) -> dict:
    """Perform basic calculations."""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero",
    }
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": operations.get(operation, "Unknown operation"),
    }


# Tool definitions
weather_tool = ToolDefinition(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters=FunctionParameters(
            type="object",
            required=["location"],
            properties={
                "location": FunctionParameter(
                    type="string",
                    description="The city and state, e.g. San Francisco, CA",
                ),
                "unit": FunctionParameter(
                    type="string",
                    description="Temperature unit (celsius or fahrenheit)",
                    allowed_values=["celsius", "fahrenheit"],
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

# Tool registry
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# Setup resource config
resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


def execute_tool_call(tool_name: str, arguments: dict) -> dict:
    """Execute a tool call and return the result."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Tool {tool_name} not found"}

    try:
        tool_fn = TOOL_REGISTRY[tool_name]
        result = tool_fn(**arguments)
        return result
    except Exception as e:
        return {"error": str(e)}


def handle_turn_with_tools(
    user_query: str,
    endpoint: AIModelEndpoint,
    messages: list[MessageItem],
    art_dir_base: str,
    max_iterations: int = 3,
) -> tuple[str, list[MessageItem]]:
    """Handle a conversation turn with tool calling support.

    Returns:
        tuple: (final_text_response, updated_messages)
    """
    current_messages = [*messages, Prompt(role="user", text=user_query)]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Note: For Anthropic, when using thinking/reasoning with tools in multi-turn conversations,
        # the assistant messages must include thinking blocks in the conversation history.
        # To simplify this example, we disable reasoning when tools are present.
        # See Anthropic docs: https://docs.claude.com/en/docs/build-with-claude/extended-thinking

        artifact_config = create_artifact_config(f"{art_dir_base}/call_{iteration}")
        client = AIModelClient(
            model_endpoint=endpoint,
            config=AIModelCallConfig(
                max_output_tokens=2000,
                max_reasoning_tokens=1024,
                reasoning_effort="low",
                reasoning=True,
                streaming=False,
                tools=[weather_tool, calculator_tool],
                tool_choice={"type": "zero_or_more"},
                artifact_config=artifact_config,
            ),
            is_async=False,
        )

        response = client.generate(messages=current_messages, instructions=["Use tools when appropriate."])

        # If provider failed or returned nothing, stop this turn gracefully
        if not response.chat_response or not response.chat_response.choices:
            # Nothing to do in this turn; return without looping further
            return "", current_messages

        choice = response.chat_response.choices[0]
        has_tool_calls = False
        text_response: str | None = None

        # Add the complete assistant response as a single message item
        # This keeps all content (text + tool calls) together as required by LLM APIs
        assistant_message = response.chat_response.to_message_item()
        if assistant_message:
            current_messages.append(assistant_message)

        # Process response contents to check for tool calls and execute them
        for content in choice.contents:
            # Check if it's a tool call
            if hasattr(content, "tool_call") and content.tool_call:
                has_tool_calls = True
                tool_call = content.tool_call

                print(f"  🔧 Tool Call: {tool_call.name}")
                print(f"     Arguments: {json.dumps(tool_call.arguments, indent=2)}")

                # Execute the tool
                tool_output = execute_tool_call(tool_call.name, tool_call.arguments)
                print(f"     Result: {json.dumps(tool_output, indent=2)}")

                # Create tool result and add to messages
                tool_result = ToolCallResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    output=tool_output,
                )
                current_messages.append(tool_result)

            # Capture text response
            else:
                # Text content can come as either plain text or a preserved OpenAI content array
                # Prefer explicit text attribute when available
                if hasattr(content, "text") and content.text:
                    text_response = content.text
                # Otherwise, try to extract from message_contents (OpenAI Responses API)
                elif hasattr(content, "message_contents") and content.message_contents:
                    try:
                        parts = content.message_contents or []
                        texts: list[str] = []
                        for p in parts:
                            if isinstance(p, dict):
                                if p.get("type") == "output_text":
                                    texts.append(p.get("text") or "")
                            else:
                                # ChatMessageContentPart
                                t = getattr(p, "type", None)
                                txt = getattr(p, "text", None)
                                if t == "output_text" and txt:
                                    texts.append(txt)
                        combined = "".join(texts).strip()
                        if combined:
                            text_response = combined
                    except Exception:
                        # Leave text_response as-is if parsing fails
                        pass

        # If no tool calls, we have the final response
        if not has_tool_calls:
            return text_response or "", current_messages

        # If we executed tool calls and we already received a text response in the same output,
        # we can return that immediately instead of spinning another iteration.
        if has_tool_calls and (text_response is not None):
            return text_response, current_messages

    return "Max iterations reached", current_messages


def run_multi_turn_with_tools():
    """Run multi-turn conversation with tool calling."""

    queries = [
        "What's the weather like in San Francisco?",
        "Great! Now calculate 15 times 23 for me.",
        "Can you add those two numbers together? (The temperature and the multiplication result)",
    ]

    messages: list[MessageItem] = []

    print("=" * 80)
    print("Multi-Turn Conversation with Tools and Messages API")
    print("=" * 80)

    # Generate a single run directory for all turns in this conversation
    run_dir = generate_run_dirname()

    for i, query in enumerate(queries):
        model_endpoint = random.choice(resource_config.model_endpoints)

        print(f"\n🔄 Turn {i + 1} with {model_endpoint.ai_model.model_name}")
        print(f"User: {query}\n")

        final_text, messages = handle_turn_with_tools(
            user_query=query,
            endpoint=model_endpoint,
            messages=messages,
            art_dir_base=f"16_tools/{run_dir}/turn_{i}",
        )

        print(f"\nAssistant: {final_text}")
        print("-" * 80)


if __name__ == "__main__":
    run_multi_turn_with_tools()
