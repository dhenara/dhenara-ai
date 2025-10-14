import datetime
import random

from include.shared_config import all_endpoints, load_resource_config

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
) -> ConversationNode:
    """Process a single conversation turn with the specified model and query."""

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,  # 128,
            reasoning_effort="low",
            streaming=False,
            reasoning=True,
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
        )

        # Display the conversation
        print(f"User: {query}")
        print(f"Model: {model_endpoint.ai_model.model_name}\n")
        print_success = False
        for content in node.response.choices[0].contents:
            print_success = print_success or content.get_text()
            print(f"Model Response Content {content.index}:\n{content.get_text()}\n")

        if not print_success:
            print(f"No Content in model response. Response is  {node.response.model_dump()}\n")

        print("-" * 80)

        # Append to nodes, so that next turn will have the context generated
        conversation_nodes.append(node)


if __name__ == "__main__":
    run_multi_turn_conversation()
