# Dhenara AI

Dhenara AI is an open source Python package for calling multiple LLM providers through one typed interface. It keeps the integration surface small: provider credentials, model selection, prompt/message formatting, streaming, tool use, and structured output all flow through the same core client types.

For full documentation, visit [docs.dhenara.com](https://docs.dhenara.com/).

## Installation

```bash
pip install dhenara-ai
```

## Credential Setup

Dhenara AI discovers credentials in one of two ways:

1. Pass an explicit file path to `ResourceConfig.load_from_file()`.
2. Set `DAI_SECRET_CONFIG_DIR` and place `dai_credentials.yaml` inside that directory.

If `DAI_SECRET_CONFIG_DIR` is unset, the default location is `/run/secrets/dai/dai_credentials.yaml`.

Use the checked-in template at `src/dhenara/ai/types/resource/credentials.yaml` as the starting point.

```bash
mkdir -p /path/to/secrets
export DAI_SECRET_CONFIG_DIR=/path/to/secrets
cp src/dhenara/ai/types/resource/credentials.yaml "$DAI_SECRET_CONFIG_DIR/dai_credentials.yaml"
```

Then edit `dai_credentials.yaml` and keep only the providers you actually use.

## Quickstart

```python
from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelAPIProviderEnum, AIModelCallConfig, ResourceConfig

resource_config = ResourceConfig()
resource_config.load_from_file(credentials_file=None, init_endpoints=True)

endpoint = resource_config.get_model_endpoint(
    model_name="claude-haiku-4-5",
    api_provider=AIModelAPIProviderEnum.ANTHROPIC,
)
if endpoint is None:
    raise RuntimeError("No Anthropic endpoint configured for claude-haiku-4-5")

client = AIModelClient(
    model_endpoint=endpoint,
    config=AIModelCallConfig(
        max_output_tokens=1024,
        reasoning=False,
        streaming=False,
    ),
    is_async=False,
)

response = client.generate(
    prompt="Explain quantum computing in simple terms.",
    instructions=["Keep the answer under 120 words."],
)

if response.chat_response:
    print(response.chat_response.text())
```

## Running The Included Examples

The example programs use the same credential-loading contract. With a secret directory configured, you can run them directly:

```bash
export DAI_SECRET_CONFIG_DIR=/path/to/secrets
python examples/14_multi_turn_with_messages_api.py
```

If you prefer an explicit credentials file, pass that path where the example calls `load_from_file()`.

## Provider SDK Surfaces

- OpenAI direct and Azure OpenAI / Microsoft Foundry OpenAI v1 use the `openai` SDK.
- Google Gemini Developer API and Gemini on Vertex AI use the `google-genai` SDK.
- Anthropic direct, Amazon Bedrock, and Anthropic on Vertex use the `anthropic` SDK.
- Amazon Bedrock configuration stays under the `amazon_bedrock` provider block. The package passes those credentials into `AnthropicBedrock`; there is no separate boto client setup in your application code.
- `microsoft_azure_ai` is not a supported text-generation surface in `dhenara-ai`; use `microsoft_openai` with an Azure OpenAI or Microsoft Foundry OpenAI v1 endpoint instead.
- For Microsoft-hosted OpenAI-compatible deployments, the `model` value in requests must be the Azure deployment name, not just the underlying vendor model family.
