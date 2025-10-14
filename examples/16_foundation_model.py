from include.shared_config import load_resource_config, openai_endpoints

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelEndpoint

# Use shared config and demonstrate cloning/overrides on OpenAI model
rc = load_resource_config()
endpoint: AIModelEndpoint = openai_endpoints(rc)[0]

client = AIModelClient(
    model_endpoint=endpoint,
    is_async=False,
)

response = client.generate(
    prompt="Tell me a joke.",
    context=[],
    instructions=[],
)

print("-" * 80)
print("Model Response:\n")
print(response.chat_response.choices[0].contents[0].get_text())
print("-" * 80)
