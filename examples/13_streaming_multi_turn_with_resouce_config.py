import datetime
import random

from include.shared_config import all_endpoints, load_resource_config

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint, ChatResponseChunk
from dhenara.ai.types.conversation import ConversationNode
from dhenara.ai.types.shared import SSEErrorResponse, SSEEventType, SSEResponse

# Initialize shared resource config and restrict to OpenAI endpoints
resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


class StreamProcessor:
    def __init__(self):
        self.previous_content_delta = None
        self.full_response_text = ""

    def process_stream_response(self, response):
        print("\nModel Response: ", end="", flush=True)
        self.full_response_text = ""

        try:
            for chunk, final_response in response.stream_generator:
                if chunk:
                    if isinstance(chunk, SSEErrorResponse):
                        self.print_error(f"{chunk.data.error_code}: {chunk.data.message}")
                        break

                    if not isinstance(chunk, SSEResponse):
                        self.print_error(f"Unknown type {type(chunk)}")
                        continue

                    if chunk.event == SSEEventType.ERROR:
                        self.print_error(f"Stream Error: {chunk}")
                        break

                    if chunk.event == SSEEventType.TOKEN_STREAM:
                        text = self.process_stream_chunk(chunk.data)
                        if text:
                            self.full_response_text += text
                        if chunk.data.done:
                            # Don't `break` as final response will be sent after this
                            pass

                if final_response:
                    return final_response
        except KeyboardInterrupt:
            self.print_warning("Stream interrupted by user")
        except Exception as e:
            self.print_error(f"Error processing stream: {e!s}")
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
                # Actual content
                text = content_delta.get_text_delta()
                if text:
                    print(f"{text}", end="", flush=True)
                    text_delta += text
        return text_delta

    def print_error(self, message: str):
        print(f"\nError: {message}")

    def print_warning(self, message: str):
        print(f"\nWarning: {message}")


def handle_streaming_conversation_turn(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    conversation_nodes: list[ConversationNode],
    stream_processor: StreamProcessor,
) -> ConversationNode:
    """Process a single conversation turn with the specified model and query, using streaming."""

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=1000,
            max_reasoning_tokens=512,  # 128,
            reasoning_effort="low",
            streaming=True,  # Enable streaming
            reasoning=True,
        ),
        is_async=False,
    )

    prompt = user_query
    context = []
    for node in conversation_nodes:
        context += node.get_context()

    # Generate streaming response
    response = client.generate(
        prompt=prompt,
        context=context,
        instructions=instructions,
    )

    # Process the streaming response
    final_response = stream_processor.process_stream_response(response)

    if not final_response:
        raise Exception("Failed to get final response from stream")

    # Create conversation node
    node = ConversationNode(
        user_query=user_query,
        input_files=[],
        response=final_response.chat_response,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return node


def run_streaming_multi_turn_conversation():
    multi_turn_queries = [
        "Tell me a short story about a robot learning to paint.",
        "Continue the story but add a twist where the robot discovers something unexpected.",
        "Conclude the story with an inspiring ending.",
    ]

    # Instructions for each turn
    instructions_by_turn = [
        ["Be creative and engaging."],
        ["Build upon the previous story seamlessly."],
        ["Bring the story to a satisfying conclusion."],
    ]

    # Create stream processor
    stream_processor = StreamProcessor()

    # Store conversation history
    conversation_nodes = []

    # Process each turn
    for i, query in enumerate(multi_turn_queries):
        # Choose a random model endpoint
        model_endpoint = random.choice(resource_config.model_endpoints)
        # OR choose if fixed order as
        # model_endpoint = resource_config.get_model_endpoint(model_name=Claude35Haiku.model_name)

        print(f"🔄 Turn {i + 1} with {model_endpoint.ai_model.model_name} from {model_endpoint.api.provider}\n")

        print(f"User: {query}")
        print(f"Model: {model_endpoint.ai_model.model_name}")

        node = handle_streaming_conversation_turn(
            user_query=query,
            instructions=instructions_by_turn[i],  # Only if you need to change instruction on each turn, else leave []
            endpoint=model_endpoint,
            conversation_nodes=conversation_nodes,
            stream_processor=stream_processor,
        )

        print("-" * 80)

        # Append to nodes, so that next turn will have the context generated
        conversation_nodes.append(node)


if __name__ == "__main__":
    run_streaming_multi_turn_conversation()
