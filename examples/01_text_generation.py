from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.genai import AIModelAPI
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude37Sonnet

# 1. Create an API
# This can be used to create multiple model endpoints for the same API provider
api = AIModelAPI(
    provider=AIModelAPIProviderEnum.ANTHROPIC,
    api_key="your_api_key",
)

# 2. Select or create an AI model
# You can either use the foundation models as it is, or create your own models
model = Claude37Sonnet

# Create the model endpoint
model_endpoint = AIModelEndpoint(api=api, ai_model=model)

# Create the client
client = AIModelClient(
    model_endpoint=model_endpoint,
    config=AIModelCallConfig(
        max_output_tokens=16000,
        reasoning=True,  # thinking/reasoning mode
        max_reasoning_tokens=8000,  # Needed only if reasoning is set
        streaming=False,
    ),
    is_async=False,  # Sync mode/ async mode
)


response = client.generate(
    prompt={
        "role": "user",
        "content": "What are three ways to improve productivity?",
    },
    context=[],  # Optional history/context. Will show this on another example
    instructions=[
        "Be specific and actionable.",  # Optional instructions
    ],
)

print(response)


# -----------------------------------------------------------------------------
# To see formatted response, use below helper fn
def print_response(response):
    for choice in response.chat_response.choices:
        for content in choice.contents:
            # Some formatting to differentiate contents
            # With reasoning=True, same response will have multiple contents
            print("-" * 80)
            print(f"Type:: {content.type}")
            print("-" * 80)
            # Actual response text
            print(f"{content.get_text()}")

    # Optionally get the usage and cost for this call.
    # Usage/Cost calculation is enabled by default, but can be disabled via setting
    print("-" * 80)
    print(f"Usage: {response.chat_response.usage}")
    print(f"Usage Charge: {response.chat_response.usage_charge}")
    print("-" * 80)


# Call response  formatting fn
print_response(response)


"""
# -----------------------------------------------------------------------------
# For Async Mode:
# NOTE: Set `is_async=True` in config above.
# Then, generate text
async def generate_text_async():
    response = await client.generate_async(
        prompt={
            "role": "user",
            "content": "Explain quantum computing to a high school student.",
        },
        context=[],
        instructions=[
            "Be concise and focus on practical applications.",
        ],
    )
    print_response(response)


# Run the async function
import asyncio

asyncio.run(generate_text_async())
"""
