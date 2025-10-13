"""Multi-turn conversation with structured output using messages API.

This example demonstrates using structured output in a multi-turn conversation
with the messages API.

Usage:
  python examples/17_multi_turn_with_structured_output_and_messages_api.py
"""

import json
import random
from typing import Literal

from pydantic import BaseModel, Field

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelAPIProviderEnum, AIModelCallConfig, AIModelEndpoint, ResourceConfig
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt

# Imports no longer needed after refactoring to use response.to_message_item()
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude35Haiku
from dhenara.ai.types.genai.foundation_models.google.chat import Gemini25Flash
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT5Nano


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
resource_config = ResourceConfig()
resource_config.load_from_file(
    credentials_file="~/.env_keys/.dhenara_credentials.yaml",
)

anthropic_api = resource_config.get_api(AIModelAPIProviderEnum.ANTHROPIC)
openai_api = resource_config.get_api(AIModelAPIProviderEnum.OPEN_AI)
google_api = resource_config.get_api(AIModelAPIProviderEnum.GOOGLE_VERTEX_AI)

resource_config.model_endpoints = [
    AIModelEndpoint(api=anthropic_api, ai_model=Claude35Haiku),
    AIModelEndpoint(api=openai_api, ai_model=GPT5Nano),
    AIModelEndpoint(api=google_api, ai_model=Gemini25Flash),
]


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
            max_output_tokens=1000,
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

    print("Assistant (Structured Output):")
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

    print("Assistant (Structured Output):")
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
        config=AIModelCallConfig(max_output_tokens=200, streaming=False),
        is_async=False,
    )

    story_response = client.generate(messages=story_messages)

    # Add the complete assistant response to messages
    assistant_message = story_response.chat_response.to_message_item()
    if assistant_message:
        messages.append(assistant_message)

    # Extract story text
    story_text = story_response.chat_response.text() or ""

    print(f"Story: {story_text}\n")

    # Now analyze it with structured output
    story_analysis, messages = handle_turn_with_structured_output(
        user_query=f"Analyze this story: {story_text}",
        endpoint=model_endpoint,
        messages=messages,
        output_schema=StoryAnalysis,
    )

    print("Story Analysis (Structured Output):")
    print(json.dumps(story_analysis.model_dump(), indent=2))
    print("-" * 80)


if __name__ == "__main__":
    run_multi_turn_with_structured_output()
