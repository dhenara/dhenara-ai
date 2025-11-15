import logging
from typing import Any, Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field

from dhenara.ai.types.genai.ai_model import (
    AIModelAPIProviderEnum,
    AIModelProviderEnum,
    ChatResponseUsage,
    UsageCharge,
)
from dhenara.ai.types.genai.dhenara.request import PromptMessageRoleEnum
from dhenara.ai.types.genai.dhenara.request.data import Content, Prompt, PromptConfig, PromptText
from dhenara.ai.types.shared.api import SSEEventType, SSEResponse
from dhenara.ai.types.shared.base import BaseModel

from ._content_items._chat_items import (
    ChatResponseContentItem,
    ChatResponseContentItemDelta,
    ChatResponseContentItemType,
    ChatResponseGenericContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCall,
    ChatResponseToolCallContentItem,
)
from ._metadata import AIModelCallResponseMetaData

logger = logging.getLogger(__name__)


class ChatResponseChoice(BaseModel):
    """A single choice/completion in the chat response"""

    index: int
    finish_reason: Any | None = None
    stop_sequence: Any | None = None
    contents: list[ChatResponseContentItem] | None = None
    metadata: dict = {}

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "index": 0,
                "contents": [
                    {
                        "role": "assistant",
                        "text": "Hello! How can I help you today?",
                    }
                ],
            }
        },
    )


class ChatResponseChoiceDelta(BaseModel):
    """A single choice/completion in the chat response"""

    index: int
    finish_reason: Any | None = None
    stop_sequence: Any | None = None
    content_deltas: list[ChatResponseContentItemDelta] | None = None
    metadata: dict = {}


class ChatResponse(BaseModel):
    """Complete chat response from an AI model

    Contains the response content, usage statistics, and provider-specific metadata
    """

    type: Literal["chat_response"] = "chat_response"
    model: str
    provider: AIModelProviderEnum
    api_provider: AIModelAPIProviderEnum | None = None
    usage: ChatResponseUsage | None = None
    usage_charge: UsageCharge | None = None
    choices: list[ChatResponseChoice] = []
    metadata: AIModelCallResponseMetaData | dict = {}
    provider_response: dict | None = Field(
        default=None,
        description=("Complete provider-native response."),
    )

    def get_visible_fields(self) -> dict:
        return self.model_dump(exclude=["choices"])

    def to_prompt(
        self,
        choice_index: int = 0,
        max_words_text: int | None = None,
    ) -> "Prompt":
        """Convert response to a context message for next turn"""

        # Get text from the first choice's contents
        if not self.choices:
            return None

        choice = self.choices[choice_index]
        if not choice.contents:
            return None

        # Combine all content items into one text
        # Filter out None values (e.g., from reasoning items with encrypted/no content)
        text_parts = [content_item.get_text() for content_item in choice.contents]
        text_parts = [part for part in text_parts if part is not None]

        text = "\n".join(text_parts)

        # Create Content object
        content = Content(type="text", text=text)

        # Create PromptText object
        prompt_text = PromptText(content=content)

        # Create and return Prompt object
        return Prompt(
            role=PromptMessageRoleEnum.ASSISTANT,
            text=prompt_text,
            config=PromptConfig(
                max_words_text=max_words_text,
                max_words_file=None,
            ),
        )

    def first(self, content_type: ChatResponseContentItemType):
        "Returns the first content of matching type"
        for choice in self.choices:
            for content in choice.contents:
                if content.type == content_type:
                    return content
        return None

    def text(self) -> str | None:
        "Returns the first text type content"
        text_item = self.first(ChatResponseContentItemType.TEXT)
        return text_item.get_text() if text_item else None  # NOTE: get_text() after introducing `message_contents`

    def reasoning(self) -> str | None:
        "Returns the first thinkning/reasoning type content"
        reasoning_item = self.first(ChatResponseContentItemType.REASONING)
        return reasoning_item.get_text() if reasoning_item else None

    def as_text(self, content_types: list[ChatResponseContentItemType | str] | None = None) -> str | None:
        "Returns the concatenated text of all matching content types"

        if content_types is None:
            content_types = ["text", "reasoning", "structured_output", "tool_call", "generic"]
        else:
            # Ensure content_types is a list
            if not isinstance(content_types, list):
                raise ValueError("content_types must be a list of ChatResponseContentItemType")
        try:
            content_types_enum = [
                ChatResponseContentItemType(ct) if isinstance(ct, str) else ct for ct in content_types
            ]
        except ValueError as e:
            raise ValueError(f"Invalid content type in content_types: {e}")

        text_parts = [
            content.get_text()
            for choice in self.choices
            for content in choice.contents
            if content.type in content_types_enum
        ]
        return "\n\n".join(text_parts) if text_parts else None

    def tools(self) -> list[ChatResponseToolCall]:
        "Returns all tool type content"
        tools = [
            content
            for choice in self.choices
            for content in choice.contents
            if content.type == ChatResponseContentItemType.TOOL_CALL
        ]
        return tools

    def structured(self) -> dict | None:
        "Returns the first structured-output type content as dict"
        structured_item = self.first(ChatResponseContentItemType.STRUCTURED_OUTPUT)
        return structured_item.structured_output.structured_data if structured_item else None

    def structured_unprocessed(self) -> dict | None:
        "Returns the first structured-output type content as dict"
        structured_item = self.first(ChatResponseContentItemType.STRUCTURED_OUTPUT)
        return structured_item.structured_output

    def structured_pyd(self) -> PydanticBaseModel:
        "Returns the first structured-output type content as its pydantic model instance configured in the input call"
        structured_item = self.first(ChatResponseContentItemType.STRUCTURED_OUTPUT)
        return structured_item.structured_output.as_pydantic() if structured_item else None

    def to_message_item(self, choice_index: int = 0) -> "ChatResponse| None":
        """Get the response choice to use as a message item in multi-turn conversations.

        This method returns the complete ChatResponsewhich contains all content items
        (text, tool calls, etc.) from the assistant's response. This preserves the proper
        message structure required by LLM providers (e.g., OpenAI requires tool calls and
        their results to be kept together).

        """
        # Return a deep copy to prevent later mutations from stripping content
        # that must be present in subsequent turns (e.g., tool calls).
        return self.model_copy(deep=True)

    def to_message_item_LEGACY(self, choice_index: int = 0) -> "ChatResponseChoice | None":  # noqa: N802
        """Get the response choice to use as a message item in multi-turn conversations.

        This method returns the complete ChatResponseChoice which contains all content items
        (text, tool calls, etc.) from the assistant's response. This preserves the proper
        message structure required by LLM providers (e.g., OpenAI requires tool calls and
        their results to be kept together).

        For OpenAI provider, includes the original provider_response in metadata to enable
        exact round-trip fidelity for multi-turn conversations.

        Args:
            choice_index: Index of the choice to return (default: 0)

        Returns:
            ChatResponseChoice if available, None otherwise
        """
        if not self.choices or choice_index >= len(self.choices):
            return None

        choice = self.choices[choice_index]

        # For OpenAI, include provider_response in choice metadata for exact round-trip
        if self.provider_response and self.provider == AIModelProviderEnum.OPEN_AI:
            # Create a copy of the choice with provider_response in metadata
            choice_copy = choice.model_copy(deep=True)
            choice_copy.metadata["provider_response"] = self.provider_response
            return choice_copy

        return choice

    @classmethod
    def from_dict(cls, data: dict) -> "ChatResponse":
        """Restore ChatResponse from a dict/JSON artifact with proper content item reconstruction.

        This method fixes deserialization issues where Pydantic's discriminated union
        creates the wrong content item class when JSON has incomplete data
        (e.g., type="reasoning" but missing thinking_summary/thinking_signature).

        Args:
            data: Dict representation of ChatResponse (e.g., from dai_response.json)

        Returns:
            ChatResponse with properly reconstructed content items
        """
        # Make a deep copy to avoid mutating the input
        data_copy = data.copy() if isinstance(data, dict) else data

        # Process choices to reconstruct content items properly
        if "choices" in data_copy:
            reconstructed_choices = []
            for choice_data in data_copy["choices"]:
                if choice_data.get("contents"):
                    reconstructed_contents = []
                    for content_data in choice_data["contents"]:
                        # Reconstruct the appropriate content item class based on available fields
                        content_item = cls._reconstruct_content_item(content_data)
                        reconstructed_contents.append(content_item)
                    choice_data["contents"] = reconstructed_contents
                reconstructed_choices.append(ChatResponseChoice(**choice_data))
            data_copy["choices"] = reconstructed_choices

        # Use standard Pydantic construction for the rest
        return cls(**data_copy)

    @staticmethod
    def _reconstruct_content_item(content_data: dict) -> ChatResponseContentItem:
        """Reconstruct the correct content item class based on available fields.

        Args:
            content_data: Dict representation of a content item

        Returns:
            Properly typed content item instance
        """
        content_type = content_data.get("type")

        item_map = {
            "text": ChatResponseTextContentItem,
            "reasoning": ChatResponseReasoningContentItem,
            "structured_output": ChatResponseStructuredOutputContentItem,
            "tool_call": ChatResponseToolCallContentItem,
            "generic": ChatResponseGenericContentItem,
        }

        item_type = item_map.get(content_type)
        if item_type:
            try:
                return item_type(**content_data)
            except Exception as e:
                logger.error(f"Error reconstructing content item of type {content_type}: {e}")
        else:
            logger.error(f"Unknown content item type: {content_type}, defaulting to Generic")

        return ChatResponseGenericContentItem(**content_data)

        # Check for reasoning-specific fields to determine if it's truly a reasoning item

    def preview_dict(self):
        """
        Returns a preview version of the response excluding the full content of choices
        but including metadata about them
        """
        _dict = self.model_dump(exclude=["choices"])

        # Add summary information about choices instead of full content
        choice_summaries = []
        for choice in self.choices:
            choice_summary = {
                "index": choice.index,
                "content_count": len(choice.contents),
                "contents_summary": [
                    {
                        "index": content.index,
                        "type": str(content.type),
                    }
                    for content in choice.contents
                ],
            }
            choice_summaries.append(choice_summary)

        _dict["choices_summary"] = choice_summaries
        return _dict


class ChatResponseChunk(BaseModel):
    """Chat response Chunk from an AI model

    Contains the response content, usage statistics, and provider-specific metadata
    """

    model: str
    provider: AIModelProviderEnum
    api_provider: AIModelAPIProviderEnum | None = None
    usage: ChatResponseUsage | None = None
    usage_charge: UsageCharge | None = None
    choice_deltas: list[ChatResponseChoiceDelta] = []
    metadata: AIModelCallResponseMetaData | dict = {}

    done: bool = Field(
        default=False,
        description="Indicates if this is the final chunk",
    )

    def get_visible_fields(self) -> dict:
        return self.model_dump(exclude=["choices"])


class StreamingChatResponse(SSEResponse[ChatResponseChunk]):
    """Specialized SSE response for chat streaming"""

    event: SSEEventType = SSEEventType.TOKEN_STREAM
    data: ChatResponseChunk
