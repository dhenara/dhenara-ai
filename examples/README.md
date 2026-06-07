# Live examples

Optionally one or more models in `openai_endpoints`, `anthropic_endpoints`, or `google_endpoints` and run run as below.

**To run one provider at a time**:

```bash
# NOTE: Its the directory containing the `dai_credentials.yaml` file, not path of that file
export DAI_SECRET_CONFIG_DIR="/path/to/scratchpad/dev/local/config/secrets"
cd dhenara_ai/examples

EXAMPLE_ACTIVE_PROVIDER=openai uv run python 19_streaming_multi_turn_structured_thinking.py
EXAMPLE_ACTIVE_PROVIDER=anthropic uv run python 19_streaming_multi_turn_structured_thinking.py
EXAMPLE_ACTIVE_PROVIDER=google uv run python 19_streaming_multi_turn_structured_thinking.py
```

**To run all providers in parallel**:

```bash
# NOTE: Its the directory containing the `dai_credentials.yaml` file, not path of that file
export DAI_SECRET_CONFIG_DIR="/path/to/scratchpad/dev/local/config/secrets"
cd dhenara_ai/examples

uv run python 19_streaming_multi_turn_structured_thinking.py
```
