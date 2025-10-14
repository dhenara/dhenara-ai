"""Multi-turn conversation using the new messages API.

This example demonstrates how to use the messages parameter instead of
the traditional prompt/context approach. The messages API provides better
type safety and follows LLM provider conventions more closely.

Usage:
  python examples/14_multi_turn_with_messages_api.py
"""

import datetime
import random

from include.console_renderer import render_response, render_usage
from include.shared_config import all_endpoints, load_resource_config

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.conversation._node import ConversationNode
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt

# Initialize shared resource config and restrict to OpenAI endpoints
resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


def handle_conversation_turn_with_messages(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    messages: list[MessageItem],
) -> ConversationNode:
    """Process a conversation turn using the messages API."""

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=False,
        ),
        is_async=False,
    )

    # Add the new user query to messages
    new_messages = [*messages, Prompt(role="user", text=user_query)]

    # Generate response using messages API
    response = client.generate(
        messages=new_messages,
        instructions=instructions,
    )

    # Create conversation node
    node = ConversationNode(
        user_query=user_query,
        input_files=[],
        response=response.chat_response,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return node


def run_multi_turn_with_messages():
    """Run a multi-turn conversation using the messages API."""

    multi_turn_queries = [
        "Tell me a short story about a robot learning to paint.",
        "Continue the story but add a twist where the robot discovers something unexpected.",
        "Conclude the story with an inspiring ending.",
    ]

    instructions_by_turn = [
        ["Be creative and engaging."],
        ["Build upon the previous story seamlessly."],
        ["Bring the story to a satisfying conclusion."],
    ]

    # Store conversation history as MessageItem list
    messages: list[MessageItem] = []

    print("=" * 80)
    print("Multi-Turn Conversation with Messages API")
    print("=" * 80)

    for i, query in enumerate(multi_turn_queries):
        model_endpoint = random.choice(resource_config.model_endpoints)

        print(f"\n🔄 Turn {i + 1} with {model_endpoint.ai_model.model_name} from {model_endpoint.api.provider}\n")

        node = handle_conversation_turn_with_messages(
            user_query=query,
            instructions=instructions_by_turn[i],
            endpoint=model_endpoint,
            messages=messages,
        )

        print(f"User: {query}")
        print(f"Model: {model_endpoint.ai_model.model_name}\n")

        # Display response using shared renderer
        render_response(node.response)

        # Display usage
        render_usage(node.response)

        print("-" * 80)

        # Build messages for next turn: user query + assistant response
        messages.append(Prompt(role="user", text=query))

        # Add the complete assistant response as a single message item
        # This keeps all content (text, tool calls, etc.) together as required by LLM APIs
        assistant_message = node.response.to_message_item()
        if assistant_message:
            messages.append(assistant_message)


if __name__ == "__main__":
    run_multi_turn_with_messages()
