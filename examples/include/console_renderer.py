"""Shared console rendering utilities for examples.

Provides consistent rendering of chat responses across all examples,
supporting both streaming and non-streaming modes with proper display
of all content types (text, reasoning, tool calls, structured output, etc.).
"""

import json
from typing import Any

from dhenara.ai.types import ChatResponse, ChatResponseChunk, ChatResponseContentItemType
from dhenara.ai.types.shared import SSEErrorResponse, SSEEventType, SSEResponse


class ConsoleColors:
    """ANSI color codes for terminal output."""

    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as pretty JSON string."""
    try:
        return json.dumps(data, indent=indent, default=str)
    except Exception:
        return str(data)


def render_content_item(content_item, choice_index: int = 0) -> str:
    """Render a single content item with appropriate formatting.

    Args:
        content_item: Content item to render (text, reasoning, tool call, etc.)
        choice_index: Index of the choice this content belongs to

    Returns:
        Formatted string representation
    """
    output = []

    content_type = content_item.type if hasattr(content_item, "type") else "unknown"
    content_index = content_item.index if hasattr(content_item, "index") else 0

    if content_type == ChatResponseContentItemType.TEXT:
        # Regular text output
        text = content_item.get_text()
        if text:
            header = f"{ConsoleColors.BOLD}📝 Text Content [{choice_index}.{content_index}]:{ConsoleColors.RESET}"
            output.append(header)
            output.append(text)

    elif content_type == ChatResponseContentItemType.REASONING:
        thinking_text = content_item.thinking_text

        thinking_summary = content_item.thinking_summary
        has_signature = content_item.thinking_signature is not None

        any_thinking = thinking_text or thinking_summary or has_signature
        if any_thinking:
            header = (
                f"{ConsoleColors.CYAN}{ConsoleColors.BOLD}🧠 Reasoning/Thinking "
                f"[{choice_index}.{content_index}]:{ConsoleColors.RESET}{ConsoleColors.CYAN}"
            )
            output.append(header)

        if thinking_text:
            output.append(thinking_text)

        if thinking_summary:
            header = (
                f"{ConsoleColors.MAGENTA}{ConsoleColors.BOLD}🔐 Reasoning Summary "
                f"[{choice_index}.{content_index}]:{ConsoleColors.RESET}{ConsoleColors.MAGENTA}"
            )
            output.append(header)
            # Handle both string and list[dict] formats
            if isinstance(thinking_summary, str):
                output.append(thinking_summary)
            elif isinstance(thinking_summary, list):
                # Extract text from list of summary dicts
                summary_texts = []
                for item in thinking_summary:
                    if isinstance(item, dict) and "text" in item:
                        summary_texts.append(item["text"])
                    else:
                        summary_texts.append(str(item))
                output.append("\n".join(summary_texts))
            else:
                output.append(str(thinking_summary))
        if has_signature:
            # Has signature but no summary text
            header = (
                f"{ConsoleColors.MAGENTA}{ConsoleColors.BOLD}🔐 Encrypted Reasoning "
                f"[{choice_index}.{content_index}]:{ConsoleColors.RESET}"
            )
            output.append(header)
            sig_type = type(content_item.thinking_signature).__name__
            output.append(f"{ConsoleColors.GRAY}  [encrypted reasoning signature: {sig_type}]{ConsoleColors.RESET}")

        if any_thinking:
            output.append(ConsoleColors.RESET)

    elif content_type == ChatResponseContentItemType.TOOL_CALL:
        # Tool call output
        header = (
            f"{ConsoleColors.YELLOW}{ConsoleColors.BOLD}🔧 Tool Call "
            f"[{choice_index}.{content_index}]:{ConsoleColors.RESET}"
        )
        output.append(header)
        if hasattr(content_item, "tool_call") and content_item.tool_call:
            tool = content_item.tool_call
            output.append(f"{ConsoleColors.YELLOW}  Function: {tool.name}{ConsoleColors.RESET}")
            if tool.arguments:
                output.append(f"{ConsoleColors.YELLOW}  Arguments:{ConsoleColors.RESET}")
                args_json = format_json(tool.arguments, indent=4)
                output.append(f"{ConsoleColors.GRAY}{args_json}{ConsoleColors.RESET}")
            if hasattr(tool, "id") and tool.id:
                output.append(f"{ConsoleColors.GRAY}  ID: {tool.id}{ConsoleColors.RESET}")
        else:
            output.append(f"{ConsoleColors.GRAY}  {content_item.get_text()}{ConsoleColors.RESET}")

    elif content_type == ChatResponseContentItemType.STRUCTURED_OUTPUT:
        # Structured output
        header = (
            f"{ConsoleColors.GREEN}{ConsoleColors.BOLD}📋 Structured Output "
            f"[{choice_index}.{content_index}]:{ConsoleColors.RESET}"
        )
        output.append(header)
        if hasattr(content_item, "structured_output") and content_item.structured_output:
            struct_out = content_item.structured_output
            if struct_out.structured_data is not None:
                data_json = format_json(struct_out.structured_data, indent=2)
                output.append(f"{ConsoleColors.GREEN}{data_json}{ConsoleColors.RESET}")
            else:
                output.append(
                    f"{ConsoleColors.RED}  Failed to parse. Raw: {struct_out.model_dump()}{ConsoleColors.RESET}"
                )
        else:
            output.append(f"{ConsoleColors.GRAY}  {content_item.get_text()}{ConsoleColors.RESET}")

    elif content_type == ChatResponseContentItemType.GENERIC:
        # Generic content
        header = (
            f"{ConsoleColors.MAGENTA}{ConsoleColors.BOLD}📦 Generic Content "
            f"[{choice_index}.{content_index}]:{ConsoleColors.RESET}"
        )
        output.append(header)
        output.append(f"{ConsoleColors.MAGENTA}{content_item.get_text()}{ConsoleColors.RESET}")

    else:
        # Unknown content type
        header = (
            f"{ConsoleColors.GRAY}{ConsoleColors.BOLD}❓ Unknown Content "
            f"[{choice_index}.{content_index}] (type={content_type}):{ConsoleColors.RESET}"
        )
        output.append(header)
        output.append(f"{ConsoleColors.GRAY}{content_item.get_text()}{ConsoleColors.RESET}")

    return "\n".join(output)


def render_response(response: ChatResponse) -> None:
    """Render a complete non-streaming chat response to console.

    Args:
        response: The complete chat response to render
    """
    if not response.choices:
        print(f"{ConsoleColors.RED}⚠️  No choices in response{ConsoleColors.RESET}")
        return

    # Render each choice
    for choice in response.choices:
        choice_index = choice.index if hasattr(choice, "index") else 0

        if len(response.choices) > 1:
            print(f"\n{ConsoleColors.BOLD}{'=' * 60}")
            print(f"Choice {choice_index}")
            print(f"{'=' * 60}{ConsoleColors.RESET}")

        if not choice.contents:
            print(f"{ConsoleColors.YELLOW}⚠️  No content in choice {choice_index}{ConsoleColors.RESET}")
            continue

        # Render each content item in the choice
        for content_item in choice.contents:
            rendered = render_content_item(content_item, choice_index)
            if rendered:
                print(rendered)
                print()  # Add spacing between content items

        # Show finish reason if available
        if hasattr(choice, "finish_reason") and choice.finish_reason:
            print(f"{ConsoleColors.GRAY}⏹️  Finish reason: {choice.finish_reason}{ConsoleColors.RESET}")


def render_usage(response: ChatResponse) -> None:
    """Render usage statistics from a response.

    Args:
        response: The response containing usage information
    """
    if not response.usage:
        print(f"{ConsoleColors.GRAY}No usage information available{ConsoleColors.RESET}")
        return

    usage = response.usage
    print(f"{ConsoleColors.BOLD}📊 Usage:{ConsoleColors.RESET} ", end="")
    print(f"{usage.prompt_tokens} prompt + {usage.completion_tokens} completion", end="")

    if usage.reasoning_tokens:
        print(f" {ConsoleColors.CYAN}(including {usage.reasoning_tokens} reasoning){ConsoleColors.RESET}", end="")

    print(f" = {usage.total_tokens} total tokens")


class StreamingRenderer:
    """Handles rendering of streaming responses to console.

    This class maintains state across streaming chunks to properly
    display all content types as they arrive.
    """

    def __init__(self):
        """Initialize the streaming renderer."""
        self.current_content_type: ChatResponseContentItemType | None = None
        self.current_choice_index: int = 0
        self.current_content_index: int = 0
        self.content_started = False
        self.full_response_text = ""

    def process_stream(self, response) -> ChatResponse | None:
        """Process a streaming response and render to console.

        Args:
            response: The streaming response object

        Returns:
            Final ChatResponse if successful, None otherwise
        """
        try:
            for chunk, final_response in response.stream_generator:
                if chunk:
                    if isinstance(chunk, SSEErrorResponse):
                        self._render_error(f"{chunk.data.error_code}: {chunk.data.message}")
                        break

                    if not isinstance(chunk, SSEResponse):
                        self._render_error(f"Unknown chunk type: {type(chunk)}")
                        continue

                    if chunk.event == SSEEventType.ERROR:
                        self._render_error(f"Stream error: {chunk}")
                        break

                    if chunk.event == SSEEventType.TOKEN_STREAM:
                        self._process_chunk(chunk.data)

                if final_response:
                    if self.content_started:
                        print()  # End line after streaming content
                    return final_response.chat_response

        except KeyboardInterrupt:
            self._render_warning("Stream interrupted by user")
        except Exception as e:
            self._render_error(f"Error processing stream: {e!s}")

        return None

    def _process_chunk(self, chunk: ChatResponseChunk) -> None:
        """Process a single streaming chunk.

        Args:
            chunk: The chunk to process
        """
        if not chunk.choice_deltas:
            return

        for choice_delta in chunk.choice_deltas:
            choice_index = choice_delta.index if hasattr(choice_delta, "index") else 0

            if not choice_delta.content_deltas:
                continue

            for content_delta in choice_delta.content_deltas:
                self._render_content_delta(content_delta, choice_index)

    def _render_content_delta(self, content_delta, choice_index: int) -> None:
        """Render a single content delta.

        Args:
            content_delta: The content delta to render
            choice_index: Index of the choice
        """
        content_type = content_delta.type if hasattr(content_delta, "type") else None
        content_index = content_delta.index if hasattr(content_delta, "index") else 0

        # Detect content type transitions
        if content_type != self.current_content_type or choice_index != self.current_choice_index:
            if self.content_started:
                # End previous content
                if self.current_content_type in (ChatResponseContentItemType.REASONING,):
                    print(ConsoleColors.RESET)  # Reset color
                print()  # New line before next content

            # Start new content section
            self._print_content_header(content_type, choice_index, content_index)
            self.current_content_type = content_type
            self.current_choice_index = choice_index
            self.current_content_index = content_index
            self.content_started = True

        # Render the actual delta content
        if content_type == ChatResponseContentItemType.TEXT:
            text_delta = content_delta.get_text_delta() if hasattr(content_delta, "get_text_delta") else None
            if text_delta:
                print(text_delta, end="", flush=True)
                self.full_response_text += text_delta

        elif content_type == ChatResponseContentItemType.REASONING:
            thinking_delta = content_delta.thinking_text_delta
            summary_delta = content_delta.thinking_summary_delta
            if thinking_delta:
                print(f"{ConsoleColors.CYAN}{thinking_delta}{ConsoleColors.RESET}", end="", flush=True)

            if summary_delta:
                print(f"{ConsoleColors.MAGENTA}{summary_delta}{ConsoleColors.RESET}", end="", flush=True)

        elif content_type == ChatResponseContentItemType.TOOL_CALL:
            # Tool calls are usually not streamed incrementally in a useful way
            # Just show that we're receiving tool call data
            print(".", end="", flush=True)

        elif content_type == ChatResponseContentItemType.STRUCTURED_OUTPUT:
            # Structured output is usually not streamed incrementally
            print(".", end="", flush=True)

        else:
            # Generic or unknown - try to get text delta
            if hasattr(content_delta, "get_text_delta"):
                text = content_delta.get_text_delta()
                if text:
                    print(text, end="", flush=True)

    def _print_content_header(
        self, content_type: ChatResponseContentItemType | None, choice_index: int, content_index: int
    ) -> None:
        """Print a header for a new content section.

        Args:
            content_type: Type of content
            choice_index: Index of the choice
            content_index: Index of the content
        """
        if content_type == ChatResponseContentItemType.TEXT:
            header = f"{ConsoleColors.BOLD}📝 Text Content [{choice_index}.{content_index}]:{ConsoleColors.RESET} "
            print(header, end="", flush=True)

        elif content_type == ChatResponseContentItemType.REASONING:
            header = (
                f"{ConsoleColors.CYAN}{ConsoleColors.BOLD}🧠 Reasoning "
                f"[{choice_index}.{content_index}]:{ConsoleColors.RESET}{ConsoleColors.CYAN} "
            )
            print(header, end="", flush=True)

        elif content_type == ChatResponseContentItemType.TOOL_CALL:
            header = (
                f"{ConsoleColors.YELLOW}{ConsoleColors.BOLD}🔧 Tool Call "
                f"[{choice_index}.{content_index}]:{ConsoleColors.RESET} "
            )
            print(header, end="", flush=True)

        elif content_type == ChatResponseContentItemType.STRUCTURED_OUTPUT:
            header = (
                f"{ConsoleColors.GREEN}{ConsoleColors.BOLD}📋 Structured Output "
                f"[{choice_index}.{content_index}]:{ConsoleColors.RESET} "
            )
            print(header, end="", flush=True)

        else:
            header = (
                f"{ConsoleColors.MAGENTA}{ConsoleColors.BOLD}📦 Content "
                f"[{choice_index}.{content_index}]:{ConsoleColors.RESET} "
            )
            print(header, end="", flush=True)

    def _render_error(self, message: str) -> None:
        """Render an error message.

        Args:
            message: Error message to display
        """
        print(f"\n{ConsoleColors.RED}{ConsoleColors.BOLD}❌ Error:{ConsoleColors.RESET} {message}")

    def _render_warning(self, message: str) -> None:
        """Render a warning message.

        Args:
            message: Warning message to display
        """
        print(f"\n{ConsoleColors.YELLOW}{ConsoleColors.BOLD}⚠️  Warning:{ConsoleColors.RESET} {message}")
