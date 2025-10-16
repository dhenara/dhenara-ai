import datetime
import random

from include.console_renderer import render_response, render_usage
from include.shared_config import all_endpoints, create_artifact_config, generate_run_dirname, load_resource_config

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.conversation._node import ConversationNode

# Initialize a shared ResourceConfig and limit to OpenAI endpoints for this run
resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


def handle_conversation_turn(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    conversation_nodes: list[ConversationNode],
    art_dir_name: str,
) -> ConversationNode:
    """Process a single conversation turn with the specified model and query."""

    artifact_config = create_artifact_config(art_dir_name)
    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=False,
            artifact_config=artifact_config,
        ),
        is_async=False,
    )

    prompt = user_query
    context = []
    for node in conversation_nodes:
        context += node.get_context()

    # Generate response
    response = client.generate(
        prompt=prompt,
        context=context,
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


def run_multi_turn_conversation():
    multi_turn_queries = [
        "Tell me a short story about a robot learning to paint.",
        "Continue the story but add a twist where the robot discovers something unexpected.",
        "Conclude the story with an inspiring ending.",
    ]

    # Instructions for each turn
    instructions_by_turn = [
        ["Be creative and engaging."],
        ["Build upon the previous story seamlessly."],
        ["Bring the story to a satisfying conclusion."],
    ]

    # Store conversation history
    conversation_nodes = []

    # Generate a single run directory for all turns in this conversation
    run_dir = generate_run_dirname()

    # Process each turn
    for i, query in enumerate(multi_turn_queries):
        # Choose a random model endpoint
        model_endpoint = random.choice(resource_config.model_endpoints)
        # OR choose if fixed order as
        # model_endpoint = resource_config.get_model_endpoint(model_name=Claude35Haiku.model_name)

        print(f"🔄 Turn {i + 1} with {model_endpoint.ai_model.model_name} from {model_endpoint.api.provider}\n")

        node = handle_conversation_turn(
            user_query=query,
            instructions=instructions_by_turn[i],  # Only if you need to change instruction on each turn, else leave []
            endpoint=model_endpoint,
            conversation_nodes=conversation_nodes,
            art_dir_name=f"12_multi/{run_dir}/iter_{i}",
        )

        # Display the conversation
        print(f"User: {query}")
        print(f"Model: {model_endpoint.ai_model.model_name}\n")

        # Render response and usage
        render_response(node.response)
        render_usage(node.response)

        print("-" * 80)

        # Append to nodes, so that next turn will have the context generated
        conversation_nodes.append(node)


if __name__ == "__main__":
    run_multi_turn_conversation()
