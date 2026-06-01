#!/usr/bin/env python3
"""Minimal hosted web-search example for a single configured endpoint."""

__test__ = False

import sys
from pathlib import Path

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig
from dhenara.ai.types.genai.dhenara.request import Prompt, WebSearchHostedTool

examples_dir = Path(__file__).resolve().parent
src_dir = examples_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from include.shared_config import create_artifact_config, load_resource_config, openai_endpoints  # noqa: E402


def main() -> None:
    resource_config = load_resource_config()
    endpoints = openai_endpoints(resource_config)
    if not endpoints:
        raise SystemExit("No OpenAI endpoints configured in the local resource config.")

    endpoint = endpoints[0]
    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            streaming=False,
            max_output_tokens=800,
            artifact_config=create_artifact_config("hosted_web_search/openai"),
            hosted_tools=[WebSearchHostedTool()],
        ),
        is_async=False,
    )

    response = client.generate(
        messages=[
            Prompt.with_text(
                "Use web search to answer this. Which city hosted the UEFA Euro 2024 final? Include at least one citation."
            )
        ]
    )

    print(response.text() or "")
    print("\nhosted_tool_usage=", getattr(response.usage, "hosted_tool_usage", None))

    for choice in response.choices:
        for item in choice.contents:
            for part in getattr(item, "message_contents", []) or []:
                if getattr(part, "annotations", None):
                    print("annotations=", part.annotations)


if __name__ == "__main__":
    main()