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

from include.console_renderer import StreamingRenderer, render_usage
from include.shared_config import all_endpoints, create_artifact_config, generate_run_dirname, load_resource_config

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint, ChatResponse, ResourceConfig
from dhenara.ai.types.conversation import ConversationNode
from dhenara.ai.types.shared import SSEErrorResponse, SSEEventType, SSEResponse


def build_resource_config() -> ResourceConfig:
    rc = load_resource_config()
    rc.model_endpoints = all_endpoints(rc)
    return rc


async def _async_process_stream(streaming_renderer: StreamingRenderer, response) -> ChatResponse | None:
    """Async wrapper that drives the shared StreamingRenderer over an async stream.

    Returns the final ChatResponse when available, or None on error/interrupt.
    """
    try:
        async for pair in response.stream_generator:
            try:
                chunk, final_resp = pair
            except Exception:
                chunk, final_resp = pair, None

            if chunk:
                if isinstance(chunk, SSEErrorResponse):
                    streaming_renderer._render_error(f"{chunk.data.error_code}: {chunk.data.message}")
                    break

                if not isinstance(chunk, SSEResponse):
                    streaming_renderer._render_error(f"Unknown chunk type: {type(chunk)}")
                    continue

                if chunk.event == SSEEventType.ERROR:
                    streaming_renderer._render_error(f"Stream error: {chunk}")
                    break

                if chunk.event == SSEEventType.TOKEN_STREAM:
                    # Delegate delta processing to the shared renderer
                    streaming_renderer._process_chunk(chunk.data)

            if final_resp:
                if streaming_renderer.content_started:
                    print()
                return final_resp.chat_response

    except KeyboardInterrupt:
        streaming_renderer._render_warning("Stream interrupted by user")
    except Exception as e:
        streaming_renderer._render_error(f"Error processing stream: {e!s}")

    return None


async def stream_turn(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    history: list[ConversationNode],
    streaming_renderer: StreamingRenderer,
    art_dir_name: str,
) -> ConversationNode:
    """Run a single async streaming turn using the shared StreamingRenderer.

    The renderer is passed in so it can be reused across turns (keeps nicer console UX).
    """
    artifact_config = create_artifact_config(art_dir_name)
    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=True,
            artifact_config=artifact_config,
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

    # Use the shared renderer via the async wrapper
    final = await _async_process_stream(streaming_renderer, response)

    if final is None:
        raise RuntimeError("Final response missing")

    node = ConversationNode(
        user_query=user_query,
        input_files=[],
        response=final,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return node


async def main():
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
    # Reuse a single StreamingRenderer across turns for consistent console output
    streaming_renderer = StreamingRenderer()

    # Generate a single run directory for all turns in this conversation
    run_dir = generate_run_dirname()

    for i, q in enumerate(queries):
        ep = random.choice(rc.model_endpoints)
        print(f"\n🔄 (Async) Turn {i + 1} with {ep.ai_model.model_name} from {ep.api.provider}\n")
        print(f"User: {q}")
        art_dir = f"13_async/{run_dir}/iter_{i}"
        node = await stream_turn(q, instructions_by_turn[i], ep, history, streaming_renderer, art_dir)
        history.append(node)

        # Display usage stats similar to the synchronous example
        try:
            render_usage(node.response)
        except Exception:
            # Don't fail the example if usage info is missing
            pass

        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
