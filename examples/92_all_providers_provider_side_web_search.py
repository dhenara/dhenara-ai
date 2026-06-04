#!/usr/bin/env python3
"""Quick provider-agnostic smoke example for hosted web search across configured providers."""

__test__ = False

import sys
from pathlib import Path

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig
from dhenara.ai.types.genai.dhenara.request import Prompt, WebSearchHostedTool

examples_dir = Path(__file__).resolve().parent
src_dir = examples_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from include.shared_config import (  # noqa: E402
    anthropic_endpoints,
    create_artifact_config,
    google_endpoints,
    load_resource_config,
    openai_endpoints,
)


def run_provider(provider_name: str, endpoints: list) -> None:
    if not endpoints:
        print(f"Skipping {provider_name}: no endpoints configured")
        return

    endpoint = endpoints[0]
    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            streaming=False,
            max_output_tokens=800,
            artifact_config=create_artifact_config(f"hosted_web_search/{provider_name.lower()}"),
            hosted_tools=[WebSearchHostedTool()],
        ),
        is_async=False,
    )

    response = client.generate(
        messages=[
            Prompt.with_text(
                "Use web search to answer this. Which city hosted the UEFA Euro 2024 final? Include concise evidence."
            )
        ]
    )

    print(f"\n{'=' * 80}")
    print(f"{provider_name}: {endpoint.ai_model.model_name}")
    print(f"{'=' * 80}")
    print(response.text() or "")
    print("hosted_tool_usage=", getattr(response.usage, "hosted_tool_usage", None))

    for choice in response.choices:
        for item in choice.contents:
            for part in getattr(item, "message_contents", []) or []:
                annotations = getattr(part, "annotations", None)
                if annotations:
                    print("annotations=", annotations)


def main() -> None:
    resource_config = load_resource_config()
    run_provider("OpenAI", openai_endpoints(resource_config))
    run_provider("Anthropic", anthropic_endpoints(resource_config))
    run_provider("Google", google_endpoints(resource_config))


if __name__ == "__main__":
    main()
