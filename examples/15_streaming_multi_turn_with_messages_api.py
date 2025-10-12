"""Streaming multi-turn conversation using the messages API.

This example demonstrates streaming responses with the messages parameter.

Usage:
  python examples/15_streaming_multi_turn_with_messages_api.py
"""

import datetime
import random

from dhenara.ai import AIModelClient
from dhenara.ai.types import (
    AIModelAPIProviderEnum,
    AIModelCallConfig,
    AIModelEndpoint,
    ChatResponseChunk,
    ResourceConfig,
)
from dhenara.ai.types.conversation import ConversationNode
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude35Haiku
from dhenara.ai.types.genai.foundation_models.google.chat import Gemini25FlashLite
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT5Nano
from dhenara.ai.types.shared import SSEErrorResponse, SSEEventType, SSEResponse

resource_config = ResourceConfig()
resource_config.load_from_file(
    credentials_file="~/.env_keys/.dhenara_credentials.yaml",
)

anthropic_api = resource_config.get_api(AIModelAPIProviderEnum.ANTHROPIC)
openai_api = resource_config.get_api(AIModelAPIProviderEnum.OPEN_AI)
google_api = resource_config.get_api(AIModelAPIProviderEnum.GOOGLE_AI)

resource_config.model_endpoints = [
    AIModelEndpoint(api=anthropic_api, ai_model=Claude35Haiku),
    AIModelEndpoint(api=openai_api, ai_model=GPT5Nano),
    AIModelEndpoint(api=google_api, ai_model=Gemini25FlashLite),
]


class StreamProcessor:
    def __init__(self):
        self.full_response_text = ""

    def process_stream_response(self, response):
        print("\nModel Response: ", end="", flush=True)
        self.full_response_text = ""

        try:
            for chunk, final_response in response.stream_generator:
                if chunk:
                    if isinstance(chunk, SSEErrorResponse):
                        print(f"\nError: {chunk.data.error_code}: {chunk.data.message}")
                        break

                    if not isinstance(chunk, SSEResponse):
                        print(f"\nError: Unknown type {type(chunk)}")
                        continue

                    if chunk.event == SSEEventType.ERROR:
                        print(f"\nStream Error: {chunk}")
                        break

                    if chunk.event == SSEEventType.TOKEN_STREAM:
                        text = self.process_stream_chunk(chunk.data)
                        if text:
                            self.full_response_text += text

                if final_response:
                    return final_response
        except KeyboardInterrupt:
            print("\nWarning: Stream interrupted by user")
        except Exception as e:
            print(f"\nError processing stream: {e!s}")
        finally:
            print("\n")
        return None

    def process_stream_chunk(self, chunk: ChatResponseChunk):
        """Process the content from a stream chunk and return extracted text"""
        text_delta = ""
        for choice_delta in chunk.choice_deltas:
            if not choice_delta.content_deltas:
                continue

            for content_delta in choice_delta.content_deltas:
                text = content_delta.get_text_delta()
                if text:
                    print(f"{text}", end="", flush=True)
                    text_delta += text
        return text_delta


def handle_streaming_turn_with_messages(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    messages: list[MessageItem],
    stream_processor: StreamProcessor,
) -> ConversationNode:
    """Process a streaming conversation turn using messages API."""

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=1000,
            max_reasoning_tokens=512,
            reasoning_effort="minimal",
            streaming=True,
            reasoning=True,
        ),
        is_async=False,
    )

    # Add new user query to messages
    new_messages = [*messages, Prompt(role="user", text=user_query)]

    # Generate streaming response using messages API
    response = client.generate(
        messages=new_messages,
        instructions=instructions,
    )

    # Process the streaming response
    final_response = stream_processor.process_stream_response(response)

    if not final_response:
        raise RuntimeError("Failed to get final response from stream")

    # Create conversation node
    node = ConversationNode(
        user_query=user_query,
        input_files=[],
        response=final_response.chat_response,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return node


def run_streaming_multi_turn_with_messages():
    """Run a streaming multi-turn conversation using messages API."""

    multi_turn_queries = [
        "Tell me a short story about a robot learning to paint.",
        "Continue the story but add a twist where the robot discovers something unexpected.",
        "Conclude the story with an inspiring ending.",
    ]

    instructions_by_turn = [
        ["Be creative and engaging."],
        ["Build upon the previous story seamlessly."],
        ["Bring the story to a satisfying conclusion."],
    ]

    stream_processor = StreamProcessor()
    messages: list[MessageItem] = []

    print("=" * 80)
    print("Streaming Multi-Turn Conversation with Messages API")
    print("=" * 80)

    for i, query in enumerate(multi_turn_queries):
        model_endpoint = random.choice(resource_config.model_endpoints)

        print(f"\n🔄 Turn {i + 1} with {model_endpoint.ai_model.model_name} from {model_endpoint.api.provider}\n")
        print(f"User: {query}")
        print(f"Model: {model_endpoint.ai_model.model_name}")

        node = handle_streaming_turn_with_messages(
            user_query=query,
            instructions=instructions_by_turn[i],
            endpoint=model_endpoint,
            messages=messages,
            stream_processor=stream_processor,
        )

        print("-" * 80)

        # Build messages for next turn
        messages.append(Prompt(role="user", text=query))
        # Append all response contents at once (use list() to create a copy)
        messages.extend(list(node.response.choices[0].contents))


if __name__ == "__main__":
    run_streaming_multi_turn_with_messages()
