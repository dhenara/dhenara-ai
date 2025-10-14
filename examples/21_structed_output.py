import datetime
import logging
import random

from include.shared_config import all_endpoints, load_resource_config
from pydantic import BaseModel, Field  # Optional dependency for examples

from dhenara.ai import AIModelClient
from dhenara.ai.types import (
    AIModelCallConfig,
    AIModelEndpoint,
    # StructuredOutputConfig,
)
from dhenara.ai.types.conversation import ConversationNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dhenara")
logger.setLevel(logging.INFO)


# Initialize all model enpoints and collect it into a ResourceConfig.
# Ideally, you will do once in your application when it boots, and make it global
resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


class ProductRatings(BaseModel):
    rating: int = Field(..., description="Rating from 1-5", ge=1, le=5)
    value_for_money_rating: int = Field(..., description="Value for money rating from 1-5", ge=1, le=5)


# Define a schema
class ProductReview(BaseModel):
    product_name: str = Field(..., description="Name of the product being reviewed")
    # rating: int = Field(..., description="Rating from 1-5", ge=1, le=5)
    rating: ProductRatings = Field(..., description="Rating")
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
            ## -- Either pass a schema using `StructuredOutputConfig`
            # structured_output=StructuredOutputConfig(
            #    output_schema=ProductReview.model_json_schema(),
            # ),
            ## -- OR pass a pydantic model
            structured_output=ProductReview,
        ),
        is_async=False,
    )

    context = []
    for node in conversation_nodes:
        context += node.get_context()

    # Generate response
    response = client.generate(
        prompt=user_query,
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
