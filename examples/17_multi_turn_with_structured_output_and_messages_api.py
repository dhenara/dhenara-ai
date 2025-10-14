"""Multi-turn conversation with structured output using messages API.

This example demonstrates using structured output in a multi-turn conversation
with the messages API.

Usage:
  python examples/17_multi_turn_with_structured_output_and_messages_api.py
"""

import json
import random
from typing import Literal

from include.console_renderer import render_response, render_usage
from include.shared_config import all_endpoints, load_resource_config
from pydantic import BaseModel, Field  # Optional dependency for examples

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt


# Define structured output schemas
class WeatherInfo(BaseModel):
    """Weather information for a location."""

    location: str = Field(description="The location name")
    temperature: float = Field(description="Temperature in Celsius")
    condition: Literal["sunny", "cloudy", "rainy", "snowy"] = Field(description="Weather condition")
    wind_speed: float = Field(description="Wind speed in km/h")


class PersonInfo(BaseModel):
    """Information about a person."""

    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Current occupation")
    hobbies: list[str] = Field(description="List of hobbies")


class StoryAnalysis(BaseModel):
    """Analysis of a story."""

    title: str = Field(description="Story title")
    main_characters: list[str] = Field(description="Main characters")
    themes: list[str] = Field(description="Major themes")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(description="Overall sentiment")
    summary: str = Field(description="Brief summary in one sentence")


# Setup resource config
resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


def handle_turn_with_structured_output(
    user_query: str,
    endpoint: AIModelEndpoint,
    messages: list[MessageItem],
    output_schema: type[BaseModel],
) -> tuple[BaseModel, list[MessageItem]]:
    """Handle a conversation turn with structured output.

    Returns:
        tuple: (structured_output, updated_messages)
    """
    current_messages = [*messages, Prompt(role="user", text=user_query)]

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=3000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=False,
            # Pass Pydantic model directly, not wrapped in StructuredOutputConfig
            structured_output=output_schema,
        ),
        is_async=False,
    )

    response = client.generate(messages=current_messages)

    if not response.chat_response or not response.chat_response.choices:
        raise RuntimeError("No response received")

    # Add the complete assistant response to messages
    # This keeps all content together as required by LLM APIs
    assistant_message = response.chat_response.to_message_item()
    if assistant_message:
        current_messages.append(assistant_message)

    # Extract structured output from response
    structured_data = response.chat_response.structured()

    if not structured_data:
        raise RuntimeError("No structured output in response")

    # Convert to Pydantic model
    structured_obj = output_schema(**structured_data)

    return structured_obj, current_messages


def run_multi_turn_with_structured_output():
    """Run multi-turn conversation with structured output."""

    print("=" * 80)
    print("Multi-Turn Conversation with Structured Output and Messages API")
    print("=" * 80)

    messages: list[MessageItem] = []

    # Turn 1: Generate weather info
    model_endpoint = random.choice(resource_config.model_endpoints)
    print(f"\n🔄 Turn 1 with {model_endpoint.ai_model.model_name}")
    print("User: Generate weather information for Paris.\n")

    weather_info, messages = handle_turn_with_structured_output(
        user_query="Generate realistic weather information for Paris, France.",
        endpoint=model_endpoint,
        messages=messages,
        output_schema=WeatherInfo,
    )

    print("📋 Structured Output:")
    print(json.dumps(weather_info.model_dump(), indent=2))
    print("-" * 80)

    # Turn 2: Generate person info (with different provider)
    model_endpoint = random.choice(resource_config.model_endpoints)
    print(f"\n🔄 Turn 2 with {model_endpoint.ai_model.model_name}")
    print("User: Create a profile for someone who lives in that city.\n")

    person_info, messages = handle_turn_with_structured_output(
        user_query=f"Create a person profile for someone who lives in {weather_info.location} "
        f"where the weather is currently {weather_info.condition}.",
        endpoint=model_endpoint,
        messages=messages,
        output_schema=PersonInfo,
    )

    print("📋 Structured Output:")
    print(json.dumps(person_info.model_dump(), indent=2))
    print("-" * 80)

    # Turn 3: Create a story and analyze it (with different provider)
    model_endpoint = random.choice(resource_config.model_endpoints)
    print(f"\n🔄 Turn 3 with {model_endpoint.ai_model.model_name}")
    print("User: Write a short story about this person and analyze it.\n")

    # First, write the story (no structured output)
    story_messages = [
        *messages,
        Prompt(
            role="user",
            text=f"Write a very short story (2-3 sentences) about {person_info.name}, "
            f"a {person_info.age}-year-old {person_info.occupation} in {weather_info.location}.",
        ),
    ]

    client = AIModelClient(
        model_endpoint=model_endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=False,
        ),
        is_async=False,
    )

    story_response = client.generate(messages=story_messages)

    # Add the complete assistant response to messages
    assistant_message = story_response.chat_response.to_message_item()
    if assistant_message:
        messages.append(assistant_message)

    # Extract story text and display with renderer
    print()
    render_response(story_response.chat_response)
    render_usage(story_response.chat_response)

    story_text = story_response.chat_response.text() or ""

    # Now analyze it with structured output
    story_analysis, messages = handle_turn_with_structured_output(
        user_query=f"Analyze this story: {story_text}",
        endpoint=model_endpoint,
        messages=messages,
        output_schema=StoryAnalysis,
    )

    print("\n📋 Story Analysis (Structured Output):")
    print(json.dumps(story_analysis.model_dump(), indent=2))
    print("-" * 80)


if __name__ == "__main__":
    run_multi_turn_with_structured_output()
