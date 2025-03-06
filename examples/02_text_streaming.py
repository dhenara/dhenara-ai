from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.genai import AIModelAPI, ChatResponseChunk
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude37Sonnet
from dhenara.ai.types.shared import SSEErrorResponse, SSEEventType, SSEResponse

# 1. Create an API
# This can be used to create multiple model endpoints for the same API provider
api = AIModelAPI(
    provider=AIModelAPIProviderEnum.ANTHROPIC,
    api_key="your_api_key",
)

# 2. Select or create an AI model
# You can either use the foundation models as it is, or create your own models
model = Claude37Sonnet

# Create the model endpoint
model_endpoint = AIModelEndpoint(api=api, ai_model=model)

# Create the client
client = AIModelClient(
    model_endpoint=model_endpoint,
    config=AIModelCallConfig(
        max_output_tokens=16000,
        reasoning=True,  # thinking/reasoning mode
        max_reasoning_tokens=8000,  # Needed only if reasoning is set
        streaming=True,
    ),
    is_async=False,  # Sync mode/ async mode
)


response = client.generate(
    prompt={
        "role": "user",
        "content": "Explain quantum computing?",
    },
    context=[],
    instructions=[],
)

print(response)


# -----------------------------------------------------------------------------
class StreamProcesor:
    def __init__(self):
        super().__init__()
        self.previous_content_delta = None

    def process_stream_response(self, response):
        print("\nAssistant: ", end="", flush=True)

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
                        self.print_stream_chunk(chunk.data)
                        if chunk.data.done:
                            # Don't `break` as final response will be send after this
                            pass

                if final_response:
                    # THe best part is that, you will get the consolidated response
                    # in the same format as you were receiving for a non-streaming case.
                    # You don't need to manage accumulating the response in different formats
                    self.print_final_response(final_response)
        except KeyboardInterrupt:
            self.print_warning("Stream interrupted by user")
        except Exception as e:
            self.print_error(f"Error processing stream: {e!s}")
        finally:
            print("\n")

    # For streaming responses
    def print_stream_chunk(self, chunk: ChatResponseChunk):
        """Print the content from a stream chunk"""
        for choice_delta in chunk.choice_deltas:
            if not choice_delta.content_deltas:
                continue

            for content_delta in choice_delta.content_deltas:
                same_content = (
                    self.previous_content_delta
                    and self.previous_content_delta.index == content_delta.index
                    and self.previous_content_delta.type == content_delta.type
                )
                if not same_content:
                    self.previous_content_delta = content_delta
                    print()
                    print("-" * 80)
                    print(f"\n[{content_delta.type}:]")
                    print("-" * 80)

                # Actual content
                text = content_delta.get_text_delta()
                if text:
                    print(f"{text}", end="", flush=True)

    def print_error(self, message: str):
        print(f"\nError: {message}")

    # This fn is same as the print_response() fn for non streaming case.
    # You will get a consolidated response as the last respone even for streaming
    def print_final_response(self, response):
        print()
        print("~" * 80)
        print("Streaming Finished. Final Consolidated Response is\n")
        print("~" * 80)
        for choice in response.chat_response.choices:
            for content in choice.contents:
                # Some formatting to differentiate contents
                # With reasoning=True, same response will have multiple contents
                print()
                print("-" * 80)
                print(f"Type:: {content.type}")
                print("-" * 80)
                # Actual response text
                print(f"{content.get_text()}")

        # Optionally get the usage and cost for this call.
        # Usage/Cost calculation is enabled by default, but can be disabled via setting
        print("-" * 80)
        print(f"Usage: {response.chat_response.usage}")
        print(f"Usage Charge: {response.chat_response.usage_charge}")
        print("-" * 80)


# -----------------------------------------------------------------------------

stream_procesor = StreamProcesor()
stream_procesor.process_stream_response(response)
