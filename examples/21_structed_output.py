import datetime
import logging
import random
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from dhenara.ai import AIModelClient
from dhenara.ai.providers.common.prompt_formatter import PromptFormatter
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint, ResourceConfig
from dhenara.ai.types.conversation._node import ConversationNode
from dhenara.ai.types.external_api import AIModelAPIProviderEnum, StructuredOutputConfig
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude35Haiku, Claude37Sonnet
from dhenara.ai.types.genai.foundation_models.google.chat import Gemini20Flash, Gemini20FlashLite
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT4oMini, O3Mini

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dhenara")
logger.setLevel(logging.INFO)


# Initialize all model enpoints and collect it into a ResourceConfig.
# Ideally, you will do once in your application when it boots, and make it global
resource_config = ResourceConfig()
resource_config.load_from_file(
    credentials_file="~/.env_keys/.dhenara_credentials.yaml",  # Path to your file
)

anthropic_api = resource_config.get_api(AIModelAPIProviderEnum.ANTHROPIC)
openai_api = resource_config.get_api(AIModelAPIProviderEnum.OPEN_AI)
google_api = resource_config.get_api(AIModelAPIProviderEnum.GOOGLE_AI)

# Create various model endpoints, and add them to resource config
resource_config.model_endpoints = [
    AIModelEndpoint(api=anthropic_api, ai_model=Claude37Sonnet),
    AIModelEndpoint(api=anthropic_api, ai_model=Claude35Haiku),
    AIModelEndpoint(api=openai_api, ai_model=O3Mini),
    AIModelEndpoint(api=openai_api, ai_model=GPT4oMini),
    AIModelEndpoint(api=google_api, ai_model=Gemini20Flash),
    AIModelEndpoint(api=google_api, ai_model=Gemini20FlashLite),
]


def get_context(previous_nodes: list[ConversationNode], destination_model: Any) -> list[Any]:
    """Process previous conversation nodes into context for the next turn."""
    context = []

    for node in previous_nodes:
        prompts = PromptFormatter.format_conversion_node_as_prompts(
            model=destination_model,
            user_query=node.user_query,
            attached_files=node.attached_files,
            previous_response=node.response,
        )
        context.extend(prompts)

    return context


# Define a schema
class ProductReview(PydanticBaseModel):
    product_name: str = Field(..., description="Name of the product being reviewed")
    rating: int = Field(..., description="Rating from 1-5", ge=1, le=5)
    pros: list[str] = Field(..., description="List of pros/positives about the product")
    cons: list[str] = Field(..., description="List of cons/negatives about the product")
    summary: str = Field(..., description="Short summary of the review")


def handle_conversation_turn(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    conversation_nodes: list[ConversationNode],
) -> ConversationNode:
    """Process a single conversation turn with the specified model and query."""
    # Create config with function calling

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=1000,
            streaming=False,
            tools=None,
            tool_choice=None,
            structured_output=StructuredOutputConfig(
                output_schema=ProductReview,
            ),
        ),
        is_async=False,
    )

    # Format the user query
    prompt = PromptFormatter.format_conversion_node_as_prompts(
        model=endpoint.ai_model,
        user_query=user_query,
        attached_files=[],
        previous_response=[],
    )[0]

    # Get context from previous turns (if any)
    context = get_context(conversation_nodes, endpoint.ai_model) if conversation_nodes else []

    # Generate response
    response = client.generate(
        prompt=prompt,
        context=context,
        instructions=instructions,
    )

    # Create conversation node
    node = ConversationNode(
        user_query=user_query,
        attached_files=[],
        response=response.chat_response,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return node


def run_multi_turn_conversation():
    multi_turn_queries = [
        "Write a review for the iPhone 15 Pro Max.",
    ]

    # Instructions for each turn
    instructions_by_turn = [""]

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
        for content in node.response.choices[0].contents:
            print(f"Model Response Content {content.index}:\n{content.get_text()}\n")
        print("-" * 80)

        # Append to nodes, so that next turn will have the context generated
        conversation_nodes.append(node)


if __name__ == "__main__":
    run_multi_turn_conversation()
