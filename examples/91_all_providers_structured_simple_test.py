#!/usr/bin/env python3
"""Quick test of structured output with reasoning for all providers."""

# This file is an interactive example script. Prevent pytest from treating it
# as part of the automated test suite.
__test__ = False

import sys
from pathlib import Path

from pydantic import BaseModel, Field

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig
from dhenara.ai.types.genai.dhenara.request import Prompt

# Add src to path
examples_dir = Path(__file__).resolve().parent
src_dir = examples_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from include.console_renderer import StreamingRenderer, render_usage  # noqa: E402
from include.shared_config import (  # noqa: E402
    anthropic_endpoints,
    create_artifact_config,
    google_endpoints,
    load_resource_config,
    openai_endpoints,
)


class TravelPlan(BaseModel):
    destination: str
    days: int = Field(ge=1, le=14)
    interests: list[str] = Field(min_length=1)


def test_provider(provider_name: str, endpoints: list):
    """Test a single provider."""
    if not endpoints:
        print(f"⏭️  Skipping {provider_name} (no endpoints configured)")
        return True

    endpoint = endpoints[0]
    print(f"\n{'=' * 80}")
    print(f"Testing {provider_name}: {endpoint.ai_model.model_name}")
    print(f"{'=' * 80}\n")

    artifact_config = create_artifact_config(f"test_struct/{provider_name.lower()}")

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=True,
            structured_output=TravelPlan,
            artifact_config=artifact_config,
        ),
        is_async=False,
    )

    renderer = StreamingRenderer()
    prompt = (
        "Create a TravelPlan JSON for a 3-day trip to Tokyo focused on food and culture. "
        "Only output JSON matching the schema exactly—no extra keys or markdown."
    )

    response = client.generate(
        messages=[Prompt(role="user", text=prompt)],
    )

    final = renderer.process_stream(response)
    if not final:
        print(f"❌ {provider_name}: No final response")
        return False

    render_usage(final)

    # Check for structured output
    if final and final.choices:
        choice = final.choices[0]
        structured_items = [c for c in choice.contents if hasattr(c, "structured_output") and c.structured_output]

        if not structured_items:
            print(f"❌ {provider_name}: No structured output items found")
            return False

        # Check the last structured item
        last_structured = structured_items[-1]
        if last_structured.structured_output.structured_data:
            print(f"\n✅ {provider_name}: Structured output parsed successfully")
            print(f"   Destination: {last_structured.structured_output.structured_data.get('destination')}")
            print(f"   Days: {last_structured.structured_output.structured_data.get('days')}")
            print(f"   Interests: {last_structured.structured_output.structured_data.get('interests')}")
            return True
        elif last_structured.structured_output.parse_error:
            print(f"❌ {provider_name}: Parse error: {last_structured.structured_output.parse_error}")
            return False
        else:
            print(f"❌ {provider_name}: No structured data and no error (unexpected)")
            return False
    else:
        print(f"❌ {provider_name}: No choices in response")
        return False


def main():
    """Test all providers."""
    resource_config = load_resource_config()

    results = {}

    # Test OpenAI
    try:
        openai_eps = openai_endpoints(resource_config)
        results["OpenAI"] = test_provider("OpenAI", openai_eps)
    except Exception as e:
        print(f"❌ OpenAI test failed with exception: {e}")
        results["OpenAI"] = False

    # Test Anthropic
    try:
        anthropic_eps = anthropic_endpoints(resource_config)
        results["Anthropic"] = test_provider("Anthropic", anthropic_eps)
    except Exception as e:
        print(f"❌ Anthropic test failed with exception: {e}")
        results["Anthropic"] = False

    # Test Google
    try:
        google_eps = google_endpoints(resource_config)
        results["Google"] = test_provider("Google", google_eps)
    except Exception as e:
        print(f"❌ Google test failed with exception: {e}")
        results["Google"] = False

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for provider, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{provider:15} {status}")

    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
