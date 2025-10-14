import os
import sys
from pathlib import Path

# Ensure local src is importable when running examples directly
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _EXAMPLES_DIR.parent / "src"
_ROOT_DIR = _EXAMPLES_DIR.parent.parent
for p in (str(_SRC_DIR), str(_ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_resource_config(credentials_file: str | None = None):
    from dhenara.ai.types import ResourceConfig

    rc = ResourceConfig()
    cred_path = credentials_file or os.environ.get("DHENARA_CREDENTIALS_FILE", "~/.env_keys/.dhenara_credentials.yaml")
    rc.load_from_file(credentials_file=cred_path)
    return rc


def openai_endpoints(rc):
    from dhenara.ai.types import AIModelAPIProviderEnum, AIModelEndpoint
    from dhenara.ai.types.genai.foundation_models.openai.chat import GPT5Nano, O3Mini

    openai_api = rc.get_api(AIModelAPIProviderEnum.OPEN_AI)
    # Single source of truth: pick the models you want to use across examples here
    return [
        AIModelEndpoint(api=openai_api, ai_model=O3Mini),
        AIModelEndpoint(api=openai_api, ai_model=GPT5Nano),
    ]


def anthropic_endpoints(rc):
    from dhenara.ai.types import AIModelAPIProviderEnum, AIModelEndpoint
    from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude35Haiku, Claude45Sonnet

    anthropic_api = rc.get_api(AIModelAPIProviderEnum.ANTHROPIC)
    # Single source of truth: pick the models you want to use across examples here
    return [
        AIModelEndpoint(api=anthropic_api, ai_model=Claude35Haiku),
        AIModelEndpoint(api=anthropic_api, ai_model=Claude45Sonnet),
    ]


def google_endpoints(rc):
    from dhenara.ai.types import AIModelAPIProviderEnum, AIModelEndpoint
    from dhenara.ai.types.genai.foundation_models.google.chat import Gemini25Flash, Gemini25FlashLite

    google_api = rc.get_api(AIModelAPIProviderEnum.GOOGLE_AI)
    # Single source of truth: pick the models you want to use across examples here
    return [
        AIModelEndpoint(api=google_api, ai_model=Gemini25Flash),
        AIModelEndpoint(api=google_api, ai_model=Gemini25FlashLite),
    ]


def all_endpoints(rc):
    return openai_endpoints(rc) + anthropic_endpoints(rc) + google_endpoints(rc)
