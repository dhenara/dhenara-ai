import datetime
import random

from include.console_renderer import StreamingRenderer, render_usage
from include.shared_config import all_endpoints, load_resource_config

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.conversation import ConversationNode

# Initialize shared resource config and restrict to OpenAI endpoints
resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


def handle_streaming_conversation_turn(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    conversation_nodes: list[ConversationNode],
    streaming_renderer: StreamingRenderer,
) -> ConversationNode:
    """Process a single conversation turn with the specified model and query, using streaming."""

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=True,
        ),
        is_async=False,
    )

    prompt = user_query
    context = []
    for node in conversation_nodes:
        context += node.get_context()

    # Generate streaming response
    response = client.generate(
        prompt=prompt,
        context=context,
        instructions=instructions,
    )

    # Process the streaming response
    final_response = streaming_renderer.process_stream(response)

    if not final_response:
        raise Exception("Failed to get final response from stream")

    # Create conversation node
    node = ConversationNode(
        user_query=user_query,
        input_files=[],
        response=final_response,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return node


def run_streaming_multi_turn_conversation():
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

    # Create streaming renderer
    streaming_renderer = StreamingRenderer()

    # Store conversation history
    conversation_nodes = []

    # Process each turn
    for i, query in enumerate(multi_turn_queries):
        # Choose a random model endpoint
        model_endpoint = random.choice(resource_config.model_endpoints)
        # OR choose if fixed order as
        # model_endpoint = resource_config.get_model_endpoint(model_name=Claude35Haiku.model_name)

        print(f"🔄 Turn {i + 1} with {model_endpoint.ai_model.model_name} from {model_endpoint.api.provider}\n")

        print(f"User: {query}")
        print(f"Model: {model_endpoint.ai_model.model_name}\n")

        node = handle_streaming_conversation_turn(
            user_query=query,
            instructions=instructions_by_turn[i],  # Only if you need to change instruction on each turn, else leave []
            endpoint=model_endpoint,
            conversation_nodes=conversation_nodes,
            streaming_renderer=streaming_renderer,
        )

        # Display usage
        render_usage(node.response)

        print("-" * 80)

        # Append to nodes, so that next turn will have the context generated
        conversation_nodes.append(node)


if __name__ == "__main__":
    run_streaming_multi_turn_conversation()
