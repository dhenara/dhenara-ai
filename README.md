# Dhenara

Dhenara is a genuinely open source Python package for interacting with various AI models in a unified way. It is a lightweight, straightforward framework for integrating multiple AI models into Python applications. It's similar in spirit to LangChain but with a focus on simplicity and minimal dependencies along with type safety using Pydantic Models.

For full documentation, visit [docs.dhenara.com](https://docs.dhenara.com/).

## Why Dhenara?

- **Genuinely Open Source**: Built from the ground up as a community resource, not an afterthought or internal tool
- **Unified API**: Interact with different AI providers through a consistent interface
- **Type Safety**: Built with Pydantic for robust type checking and validation
- **Easy Regeneration across Providers**: With a unified Pydantic output and built-in prompt formatting, send output from a model to any other model easily
- **Streaming**: First-class support for streaming responses along with accumulated responses similar to non-streaming responses
- **Async Support**: Both synchronous and asynchronous interfaces for maximum flexibility
- **Resource Management**: Automatic handling of connections, retries, and timeouts
- **Foundation Models**: Pre-configured models with sensible defaults
- **Test Mode**: Bring up your app with dummy responses for streaming and non-streaming generation
- **Cost/Usage Data**: Derived cost and usage data along with responses, with optional charge for each model endpoint for commercial deployment
- **Community-Oriented Design**: An architecture separating API credentials, models, and configurations for flexible deployment and scaling

## Example Usage

Here's a simple example of using Dhenara to interact with an AI model. You can find more examples in [docs.dhenara.com](https://docs.dhenara.com/).

```python
from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.genai import AIModelAPI
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude37Sonnet

# Create an API
api = AIModelAPI(
    provider=AIModelAPIProviderEnum.ANTHROPIC,
    api_key="your_api_key",
)

# Create an endpoint using a pre-configured model
model_endpoint = AIModelEndpoint(
    api=api,
    ai_model=Claude37Sonnet,
)

# Configure the api call
config = AIModelCallConfig(
    max_output_tokens=16000,
    reasoning=True,  # Thinking/reasoning mode
    max_reasoning_tokens=8000,
    streaming=False,
)

# Create the client
client = AIModelClient(
    model_endpoint=model_endpoint,
    config=config,
    is_async=False,
)

# Create a prompt
prompt = {
    "role": "user",
    "content": "Explain quantum computing in simple terms",
}

# Generate a response
response = client.generate(prompt=prompt)

# If not streaming
if response.chat_response:
    print(response.chat_response.choices[0].contents[0].get_text())

# If streaming
elif response.stream_generator:
    for chunk, _ in response.stream_generator:
        if chunk:
            print(
                chunk.data.choice_deltas[0].content_deltas[0].get_text_delta(),
                end="",
                flush=True,
            )
```

## Documentation

For full documentation, visit [docs.dhenara.com](https://docs.dhenara.com/).