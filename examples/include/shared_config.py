from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from dhenara.ai.types import ResourceConfig
from dhenara.ai.types.genai.dhenara.request import ArtifactConfig

# Global switch to enable/disable artifacts for all examples
# Set to False to disable artifact generation across all examples
ENABLE_ARTIFACTS = True

# Ensure local src is importable when running examples directly
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _EXAMPLES_DIR.parent / "src"
_ROOT_DIR = _EXAMPLES_DIR.parent.parent
for p in (str(_SRC_DIR), str(_ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_resource_config(credentials_file: str | None = None):
    rc = ResourceConfig()
    rc.load_from_file(credentials_file)
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
    from dhenara.ai.types.genai.foundation_models.anthropic.chat import ClaudeHaiku45, ClaudeSonnet45

    anthropic_api = rc.get_api(AIModelAPIProviderEnum.ANTHROPIC)
    # Single source of truth: pick the models you want to use across examples here
    return [
        AIModelEndpoint(api=anthropic_api, ai_model=ClaudeHaiku45),
        AIModelEndpoint(api=anthropic_api, ai_model=ClaudeSonnet45),
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
    # Enable all providers by default; uncomment a single line below to focus one provider while debugging
    # return openai_endpoints(rc)
    # return anthropic_endpoints(rc)
    # return google_endpoints(rc)
    return openai_endpoints(rc) + anthropic_endpoints(rc) + google_endpoints(rc)


def generate_run_dirname() -> str:
    """Generate a run directory name with timestamp.

    Returns:
        Directory name in format: run_{YYYYMMDD_HHMMSS}

    Example:
        >>> generate_run_dirname()
        'run_20251016_093239'
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"


def create_artifact_config(dir_name: str) -> ArtifactConfig | None:
    """Create artifact configuration for an example.

    Args:
        dir_name: Directory path relative to dai_artifacts/
                  (e.g., "14_multi/run_20251016_093239/iter_0")

    Returns:
        ArtifactConfig with path: dai_artifacts/{dir_name}/ if ENABLE_ARTIFACTS is True,
        None otherwise (disables artifact generation)

    Example:
        >>> run_dir = generate_run_dirname()
        >>> config = create_artifact_config(f"14_multi/{run_dir}/iter_0")
        >>> # Creates: dai_artifacts/14_multi/run_20251016_093239/iter_0/
        >>> # Or returns None if ENABLE_ARTIFACTS = False
    """
    # Check global switch - return None to disable artifacts
    if not ENABLE_ARTIFACTS:
        return None

    artifact_root = _EXAMPLES_DIR / "dai_artifacts" / dir_name

    return ArtifactConfig(
        enabled=True,
        artifact_root=str(artifact_root),
        capture_dhenara_request=True,
        capture_provider_request=True,
        capture_provider_response=True,
        capture_dhenara_response=True,
        prefix=None,
        enable_python_logs=True,
        python_log_level="DEBUG",
        python_logger_levels={
            "httpcore": "WARNING",
            "httpx": "WARNING",
            "urllib3": "WARNING",
            "dhenara.ai": "DEBUG",
        },
    )
