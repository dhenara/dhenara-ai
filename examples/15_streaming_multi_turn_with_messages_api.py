"""Streaming multi-turn conversation using the messages API.

This example demonstrates streaming responses with the messages parameter.

Usage:
  python examples/15_streaming_multi_turn_with_messages_api.py
"""

import datetime
import random

from include.console_renderer import StreamingRenderer, render_usage
from include.shared_config import all_endpoints, create_artifact_config, generate_run_dirname, load_resource_config

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.conversation import ConversationNode
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt

resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


def handle_streaming_turn_with_messages(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    messages: list[MessageItem],
    streaming_renderer: StreamingRenderer,
    art_dir_name: str,
) -> ConversationNode:
    """Process a streaming conversation turn using messages API."""

    # Create artifact config for this example to capture request/response artifacts
    artifact_config = create_artifact_config(art_dir_name)

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=True,
            artifact_config=artifact_config,
        ),
        is_async=False,
    )

    # Add new user query to messages
    new_messages = [*messages, Prompt(role="user", text=user_query)]

    # Generate streaming response using messages API
    response = client.generate(
        messages=new_messages,
        instructions=instructions,
    )

    # Process the streaming response
    final_response = streaming_renderer.process_stream(response)

    if not final_response:
        raise RuntimeError("Failed to get final response from stream")

    # Create conversation node
    node = ConversationNode(
        user_query=user_query,
        input_files=[],
        response=final_response,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return node


def run_streaming_multi_turn_with_messages():
    """Run a streaming multi-turn conversation using messages API."""

    multi_turn_queries = [
        "Tell me a short story about a robot learning to paint under 500 words.",
        "Continue the story but add a twist where the robot discovers something unexpected.",
        "Conclude the story with an inspiring ending.",
    ]

    instructions_by_turn = [
        ["Be creative and engaging."],
        ["Build upon the previous story seamlessly."],
        ["Bring the story to a satisfying conclusion."],
    ]

    streaming_renderer = StreamingRenderer()
    messages: list[MessageItem] = []

    print("=" * 80)
    print("Streaming Multi-Turn Conversation with Messages API")
    print("=" * 80)

    # Generate a single run directory for all turns in this conversation
    run_dir = generate_run_dirname()

    for i, query in enumerate(multi_turn_queries):
        model_endpoint = random.choice(resource_config.model_endpoints)

        print(f"\n🔄 Turn {i + 1} with {model_endpoint.ai_model.model_name} from {model_endpoint.api.provider}\n")
        print(f"User: {query}")
        print(f"Model: {model_endpoint.ai_model.model_name}\n")

        node = handle_streaming_turn_with_messages(
            user_query=query,
            instructions=instructions_by_turn[i],
            endpoint=model_endpoint,
            messages=messages,
            streaming_renderer=streaming_renderer,
            art_dir_name=f"15_streaming/{run_dir}/iter_{i}",
        )

        # Display usage
        render_usage(node.response)

        print("-" * 80)

        # Build messages for next turn
        messages.append(Prompt(role="user", text=query))
        # Add the complete assistant response as a single message item
        assistant_message = node.response.to_message_item()
        if assistant_message:
            messages.append(assistant_message)


if __name__ == "__main__":
    run_streaming_multi_turn_with_messages()
