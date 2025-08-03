import datetime
import random

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelAPIProviderEnum, AIModelCallConfig, AIModelEndpoint, ResourceConfig
from dhenara.ai.types.conversation._node import ConversationNode
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude35Haiku, Claude40Sonnet
from dhenara.ai.types.genai.foundation_models.google.chat import Gemini25Flash, Gemini25FlashLite
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT41Nano, O3Mini

# Initialize all model enpoints and collect it into a ResourceConfig.
# Ideally, you will do once in your application when it boots, and make it global
resource_config = ResourceConfig()
resource_config.load_from_file(
    credentials_file="~/.env_keys/.dhenara_credentials.yaml",  # Path to your file
)

anthropic_api = resource_config.get_api(AIModelAPIProviderEnum.ANTHROPIC)
openai_api = resource_config.get_api(AIModelAPIProviderEnum.OPEN_AI)
google_api = resource_config.get_api(AIModelAPIProviderEnum.GOOGLE_VERTEX_AI)

# Create various model endpoints, and add them to resource config
resource_config.model_endpoints = [
    AIModelEndpoint(api=anthropic_api, ai_model=Claude40Sonnet),
    AIModelEndpoint(api=anthropic_api, ai_model=Claude35Haiku),
    AIModelEndpoint(api=openai_api, ai_model=O3Mini),
    AIModelEndpoint(api=openai_api, ai_model=GPT41Nano),
    AIModelEndpoint(api=google_api, ai_model=Gemini25Flash),
    AIModelEndpoint(api=google_api, ai_model=Gemini25FlashLite),
]


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
            max_output_tokens=1000,
            max_reasoning_tokens=512,  # 128,
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
