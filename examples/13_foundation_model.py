from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelEndpoint
from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.genai import AIModelAPI, ChatModelCostData
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude35Sonnet

# Initialize API configurations
bedrock_api = AIModelAPI(
    provider=AIModelAPIProviderEnum.AMAZON_BEDROCK,
    credentials={
        "access_key_id": "your_access_key_id",
        "secret_access_key": "your_secret_access_key",
    },
    config={"region": "your_region"},
)


bedrock_claude35_sonnet = Claude35Sonnet.clone(
    model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
)
# NOTE: Remove the `-latest` version_suffix set in Claude35Sonnet
bedrock_claude35_sonnet.metadata["version_suffix"] = None

# Optionally override settings/options/cost-data if you need
bedrock_claude35_sonnet.cost_data = ChatModelCostData(
    input_token_cost_per_million=3.0,
    output_token_cost_per_million=15.0,
)

# Create Model-Endpoint
model_endpoint = AIModelEndpoint(
    api=bedrock_api,
    ai_model=bedrock_claude35_sonnet,
)


# Create the client
client = AIModelClient(
    model_endpoint=model_endpoint,
    is_async=False,
)


response = client.generate(
    prompt={
        "role": "user",
        "content": "Tell me a joke.",
    },
    context=[],
    instructions=[],
)

print("-" * 80)
print("Model Response:\n")
print(response.chat_response.choices[0].contents[0].get_text())
print("-" * 80)
