"""Async streaming multi-turn conversation example.

Mirrors the synchronous example (13_streaming_multi_turn_with_resouce_config.py)
but uses `is_async=True` client and awaits `generate_async`.

Usage:
  python examples/13_async_streaming_multi_turn_with_resouce_config.py
"""
from __future__ import annotations

import asyncio
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
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude35Haiku
from dhenara.ai.types.genai.foundation_models.google.chat import Gemini25FlashLite
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT4oMini
from dhenara.ai.types.shared import SSEErrorResponse, SSEEventType, SSEResponse


def build_resource_config() -> ResourceConfig:
    rc = ResourceConfig()
    rc.load_from_file(credentials_file="~/.env_keys/.dhenara_credentials.yaml")
    anthropic_api = rc.get_api(AIModelAPIProviderEnum.ANTHROPIC)
    openai_api = rc.get_api(AIModelAPIProviderEnum.OPEN_AI)
    google_api = rc.get_api(AIModelAPIProviderEnum.GOOGLE_AI)
    rc.model_endpoints = [
        AIModelEndpoint(api=anthropic_api, ai_model=Claude35Haiku),
        AIModelEndpoint(api=openai_api, ai_model=GPT4oMini),
        AIModelEndpoint(api=google_api, ai_model=Gemini25FlashLite),
    ]
    return rc


async def stream_turn(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    history: list[ConversationNode],
) -> ConversationNode:
    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=1000,
            max_reasoning_tokens=512,
            streaming=True,
            reasoning=True,
        ),
        is_async=True,
    )
    context: list[str] = []
    for node in history:
        context += node.get_context()

    response = await client.generate_async(
        prompt=user_query,
        context=context,
        instructions=instructions,
    )

    print("\nModel Response: ", end="", flush=True)
    final = None
    async for pair in response.stream_generator:  # pair expected (chunk, final_resp)
        try:
            chunk, final_resp = pair
        except Exception:
            # Defensive: some implementations may yield already-unpacked
            chunk, final_resp = pair, None
        if chunk:
            if isinstance(chunk, SSEErrorResponse):
                print(f"\n[Error] {chunk.data.error_code}: {chunk.data.message}")
                break
            if isinstance(chunk, SSEResponse):
                if chunk.event == SSEEventType.ERROR:
                    print(f"\n[Error] {chunk}")
                    break
                if chunk.event == SSEEventType.TOKEN_STREAM and isinstance(chunk.data, ChatResponseChunk):
                    for choice_delta in chunk.data.choice_deltas:
                        for content_delta in choice_delta.content_deltas or []:
                            text = content_delta.get_text_delta() or ""
                            if text:
                                print(text, end="", flush=True)
        if final_resp:
            final = final_resp
            break
    print("\n")
    if final is None:
        raise RuntimeError("Final response missing")
    node = ConversationNode(
        user_query=user_query,
        input_files=[],
        response=final.chat_response,
        timestamp=datetime.datetime.now().isoformat(),
    )
    return node


async def main():  # pragma: no cover - example script
    rc = build_resource_config()
    history: list[ConversationNode] = []
    queries = [
        "Tell me a short story about a robot learning to paint.",
        "Continue the story but add a twist where the robot discovers something unexpected.",
        "Conclude the story with an inspiring ending.",
    ]
    instructions_by_turn = [
        ["Be creative and engaging."],
        ["Build upon the previous story seamlessly."],
        ["Bring the story to a satisfying conclusion."],
    ]
    for i, q in enumerate(queries):
        ep = random.choice(rc.model_endpoints)
        print(f"\n🔄 (Async) Turn {i+1} with {ep.ai_model.model_name} from {ep.api.provider}\n")
        print(f"User: {q}")
        node = await stream_turn(q, instructions_by_turn[i], ep, history)
        history.append(node)
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
