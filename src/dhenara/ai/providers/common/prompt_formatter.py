import logging

from dhenara.ai.providers.anthropic import AnthropicClientBase
from dhenara.ai.providers.google import GoogleAIClientBase
from dhenara.ai.providers.openai import OpenAIClientBase
from dhenara.ai.types.external_api import (
    AIModelProviderEnum,
    AnthropicMessageRoleEnum,
    FormattedPrompt,
    GoogleAiMessageRoleEnum,
    OpenAiMessageRoleEnum,
)
from dhenara.ai.types.genai import AIModel, ChatResponse, ChatResponseChoice, ImageResponse
from dhenara.ai.types.shared.file import GenericFile

logger = logging.getLogger(__name__)


class PromptFormatter:
    """Handles conversion of content into model-specific prompt formats"""

    PROVIDER_CONFIG = {
        AIModelProviderEnum.OPEN_AI: {
            "user_role": OpenAiMessageRoleEnum.USER,
            "assistant_role": OpenAiMessageRoleEnum.ASSISTANT,
            "formatter": OpenAIClientBase,
        },
        AIModelProviderEnum.GOOGLE_AI: {
            "user_role": GoogleAiMessageRoleEnum.USER,
            "assistant_role": GoogleAiMessageRoleEnum.MODEL,
            "formatter": GoogleAIClientBase,
        },
        AIModelProviderEnum.ANTHROPIC: {
            "user_role": AnthropicMessageRoleEnum.USER,
            "assistant_role": AnthropicMessageRoleEnum.ASSISTANT,
            "formatter": AnthropicClientBase,
        },
        AIModelProviderEnum.DEEPSEEK: {
            "user_role": OpenAiMessageRoleEnum.USER,  # Note OpenAI roles
            "assistant_role": OpenAiMessageRoleEnum.ASSISTANT,
            "formatter": OpenAIClientBase,
        },
    }

    @staticmethod
    def _truncate_text_by_words(text: str, max_words: int | None) -> str:
        """Helper method to truncate text by word count"""
        if not max_words or not text:
            return text
        words = text.split()
        return " ".join(words[:max_words])

    @staticmethod
    def format_conversion_node_as_prompts(
        model: AIModel,
        user_query: str | None = None,
        attached_files: list[GenericFile] | None = None,
        previous_response: ChatResponse | ImageResponse | None = None,
        max_words_query: int | None = None,
        max_words_files: int | None = None,
        max_words_response: int | None = None,
        response_before_query: bool = False,
        concat_previous_response_content_items: bool = True,
    ) -> list[FormattedPrompt]:
        """
        Formats conversation elements into model-specific prompts.

        Args:
            model: The AI model being used
            user_query: The user's question or input
            attached_files: list of files to be included in the prompt
            previous_response: Previous model response to include in conversation
            max_words_query: Maximum words for user query
            max_words_files: Maximum words for file content
            max_words_response: Maximum words for previous response
            response_before_query: Whether to place response before query in prompt

        Returns:
            list of formatted prompt messages
        """
        # Extract and truncate previous response if available
        response_messages = []
        _previous_response_text = None
        config = PromptFormatter.PROVIDER_CONFIG[model.provider]
        formatter = config["formatter"]

        if attached_files is None:
            attached_files = []

        if (attached_files and not isinstance(attached_files, list)) or not all(
            isinstance(f, GenericFile) for f in attached_files
        ):
            raise ValueError(f"Invalid type {type(attached_files)} for attached files. Should be list of GenericFile")

        if previous_response:
            try:
                if isinstance(previous_response, dict):
                    _choices = previous_response.get("choices", [])
                    if _choices:
                        choices = [ChatResponseChoice(**choice) for choice in _choices]
                elif isinstance(previous_response, (ChatResponse, ImageResponse)):
                    choices = previous_response.choices
                else:
                    raise ValueError(
                        f"format_as_prompts: previous_response type {type(previous_response)} not supported"
                    )

                _previous_content_items = [content_item for choice in choices for content_item in choice.contents]

                # NOTE: Output files generated in previous response are not handled.
                # They need to be passed in as attached_files to be included in the prompt.
                # TODO: Add docs
                if concat_previous_response_content_items:
                    response_text = None
                    _previous_response_text = " ".join(
                        [
                            f"type:{content_item.type}, content: {content_item.get_text()}"
                            for content_item in _previous_content_items
                        ]
                    )

                    if _previous_response_text:
                        response_text = PromptFormatter._truncate_text_by_words(
                            _previous_response_text, max_words_response
                        )
                    # Format previous response if present
                    if response_text:
                        response_message = formatter.get_prompt(
                            model=model,
                            role=config["assistant_role"],
                            text=response_text,
                            file_contents=[],
                        )
                        response_messages = [response_message]

                else:
                    # TODO_FUTURE: max_words_response has no effect here
                    response_messages = [
                        formatter.get_prompt(
                            model=model,
                            role=config["assistant_role"],
                            text=content_item.get_text(),
                            file_contents=[],
                        )
                        for content_item in _previous_content_items
                    ]

            except Exception as e:
                logger.error(f"Failed to extract previous response text: {e}")

        # Truncate user query if needed
        query_text = PromptFormatter._truncate_text_by_words(user_query, max_words_query) if user_query else None

        # Initialize prompt components
        file_contents = []
        query_message = None

        # Validate provider
        if model.provider not in PromptFormatter.PROVIDER_CONFIG:
            raise ValueError(f"Unsupported model provider: {model.provider}")

        # Format query if present
        if query_text:
            if attached_files:
                file_contents = formatter.get_prompt_file_contents(
                    model=model,
                    files=attached_files,
                    max_words=max_words_files,
                )
            query_message = formatter.get_prompt(
                model=model,
                role=config["user_role"],
                text=query_text,
                file_contents=file_contents,
            )

        # Arrange messages in requested order
        messages = []
        if response_before_query:
            if response_messages:
                messages.extend(response_messages)
            if query_message:
                messages.append(query_message)
        else:
            if query_message:
                messages.append(query_message)
            if response_messages:
                messages.extend(response_messages)

        return messages
