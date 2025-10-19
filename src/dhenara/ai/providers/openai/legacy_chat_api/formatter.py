import logging
from typing import Any

from dhenara.ai.providers.base import BaseFormatter
from dhenara.ai.providers.openai.message_converter import OpenAIMessageConverter
from dhenara.ai.types.genai.ai_model import AIModelEndpoint, AIModelFunctionalTypeEnum
from dhenara.ai.types.genai.dhenara.request import (
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    MessageItem,
    Prompt,
    PromptMessageRoleEnum,
    StructuredOutputConfig,
    ToolCallResult,
    ToolCallResultsMessage,
    ToolChoice,
    ToolDefinition,
)
from dhenara.ai.types.genai.dhenara.request.data import FormattedPrompt
from dhenara.ai.types.genai.dhenara.response import ChatResponseChoice
from dhenara.ai.types.shared.file import FileFormatEnum, GenericFile

logger = logging.getLogger(__name__)


class OpenAIFormatterCHATAPI(BaseFormatter):
    """
    Formatter for converting Dhenara types to OpenAI-specific formats and vice versa.
    """

    role_map = {
        PromptMessageRoleEnum.USER: "user",
        PromptMessageRoleEnum.ASSISTANT: "assistant",
        PromptMessageRoleEnum.SYSTEM: "system",
    }

    @classmethod
    def convert_prompt(
        cls,
        formatted_prompt: FormattedPrompt,
        model_endpoint: AIModelEndpoint | None = None,
        files: list[GenericFile] | None = None,
        max_words_file: int | None = None,
    ) -> dict[str, Any]:
        # Map Dhenara formats to provider format
        file_contents = None
        if files:
            file_contents = cls.convert_files_to_provider_content(
                files=files,
                model_endpoint=model_endpoint,
                max_words=max_words_file,
            )

        if model_endpoint.ai_model.functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
            return cls._convert_image_model_prompt(
                formatted_prompt=formatted_prompt,
                model_endpoint=model_endpoint,
                file_contents=file_contents,
            )

        if file_contents:
            content = [
                {
                    "type": "text",
                    "text": formatted_prompt.text,
                },
                *file_contents,
            ]
        else:
            # Use Simple text formtat
            content = formatted_prompt.text

        role = cls.role_map.get(formatted_prompt.role)
        return {"role": role, "content": content}

    @classmethod
    def convert_instruction_prompt(
        cls,
        formatted_prompt: FormattedPrompt,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        # Beta models won't support System role
        if model_endpoint.ai_model.beta:
            role = cls.role_map.get(PromptMessageRoleEnum.USER)
        else:
            role = cls.role_map.get(formatted_prompt.role)

        return {"role": role, "content": formatted_prompt.text}

    @classmethod
    def convert_files_to_provider_content(
        cls,
        files: list[GenericFile],
        model_endpoint: AIModelEndpoint | None = None,
        max_words: int | None = None,
    ) -> list[dict[str, Any]]:
        if model_endpoint.ai_model.functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
            return cls._convert_files_for_image_models(
                files=files,
                model_endpoint=model_endpoint,
                max_words=max_words,
            )

        contents = []
        for file in files:
            file_format = file.get_file_format()
            try:
                if file_format in [FileFormatEnum.COMPRESSED, FileFormatEnum.TEXT]:
                    text = f"\nFile: {file.get_source_file_name()}  Content: {file.get_processed_file_data(max_words)}"
                    pcontent = {
                        "type": "text",
                        "text": text,
                    }
                    contents.append(pcontent)
                elif file_format in [FileFormatEnum.IMAGE]:
                    mime_type = file.get_mime_type()
                    if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                        raise ValueError(f"Unsupported media type: {mime_type} for file {file.name}")

                    data_content = file.get_processed_file_data_content_only()
                    pcontent = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{data_content}",
                        },
                    }
                    contents.append(pcontent)
                else:
                    logger.error(f"convert_file_content: Unknown file_format {file_format} for file {file.name} ")
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {e}")

        return contents

    # -------------------------------------------------------------------------
    # Internal helper fn
    @classmethod
    def _convert_image_model_prompt(
        cls,
        formatted_prompt: FormattedPrompt,
        file_contents: list[dict[str, Any]],
        model_endpoint: AIModelEndpoint | None = None,
    ) -> str:
        if file_contents:
            _file_content = " ".join(file_contents)
            content = formatted_prompt.text + " " + _file_content
        else:
            content = formatted_prompt.text

        return content

    @classmethod
    def _convert_files_for_image_models(
        cls,
        files: list[GenericFile],
        model_endpoint: AIModelEndpoint | None = None,
        max_words: int | None = None,
    ) -> str:
        contents: list[dict[str, Any]] = []
        for file in files:
            file_format = file.get_file_format()
            try:
                if file_format in [FileFormatEnum.COMPRESSED, FileFormatEnum.TEXT]:
                    text = (
                        f"\nFile: {file.get_source_file_name()}  "
                        f"Content: {file.get_processed_file_data(max_words=max_words)}"
                    )
                    contents.append(text)
                elif file_format == FileFormatEnum.IMAGE:
                    mime_type = file.get_mime_type()
                    if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                        raise ValueError(f"Unsupported media type: {mime_type} for file {file.name}")

                    data_content = file.get_processed_file_data_content_only()
                    pcontent = f"data:{mime_type};base64,{data_content}"
                    contents.append(pcontent)
                else:
                    logger.error(
                        f"_convert_files_for_image_models: Unknown file_format {file_format} for file {file.name}"
                    )
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {e}")

        return " ".join(contents)

    # -------------------------------------------------------------------------

    # Tools & Structured output
    @classmethod
    def convert_function_parameter(
        cls,
        param: FunctionParameter,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionParameter to OpenAI format"""
        # Drop None-valued fields (OpenAI rejects nulls for schema keys like description)
        result = param.model_dump(
            exclude={"required", "allowed_values", "default"},
            exclude_none=True,
        )
        return result

    @classmethod
    def convert_function_parameters(
        cls,
        params: FunctionParameters,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionParameters to OpenAI format"""
        # Create a new dictionary with transformed properties
        result: dict[str, Any] = {
            "type": params.type,
            "properties": {name: cls.convert_function_parameter(param) for name, param in params.properties.items()},
        }

        # Be explicit to avoid tool schemas being too permissive
        result["additionalProperties"] = False

        # Auto-build the required list based on parameters marked as required
        required_params = [name for name, param in params.properties.items() if param.required]

        # Only include required field if there are required parameters
        if required_params:
            result["required"] = required_params
        elif params.required:  # If manually specified required array exists
            result["required"] = params.required

        return result

    @classmethod
    def convert_function_definition(
        cls,
        func_def: FunctionDefinition,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionDefinition to OpenAI format"""
        res = {
            "name": func_def.name,
            "parameters": cls.convert_function_parameters(func_def.parameters),
        }
        # Only include description if present and non-empty
        if getattr(func_def, "description", None):
            res["description"] = func_def.description
        return res

    @classmethod
    def convert_tool(
        cls,
        tool: ToolDefinition,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> Any:
        """Convert ToolDefinition to OpenAI format"""
        return {
            "type": "function",
            "function": cls.convert_function_definition(tool.function),
        }

    @classmethod
    def convert_tool_choice(
        cls,
        tool_choice: ToolChoice,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> Any:
        """Convert ToolChoice to OpenAI format"""
        if tool_choice is None:
            return None

        if tool_choice.type is None:
            return None
        elif tool_choice.type == "zero_or_more":
            return "auto"
        elif tool_choice.type == "one_or_more":
            return "required"
        elif tool_choice.type == "specific":
            return {"type": "function", "name": tool_choice.specific_tool_name}

    @classmethod
    def convert_structured_output(
        cls,
        structured_output: StructuredOutputConfig,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert StructuredOutputConfig to OpenAI format"""
        # Get the original JSON schema from Pydantic or dict
        schema = structured_output.get_schema()

        # Ensure additionalProperties is set in the root schema and all nested definitions
        schema["additionalProperties"] = False

        # Also set additionalProperties for nested schemas
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                def_schema["additionalProperties"] = False

        # Extract the name from the title and use it for schema name
        schema_name = schema.get("title", "output")

        # Clean up JSON Schema keywords that OpenAI doesn't permit
        if "properties" in schema:
            for prop in schema["properties"].values():
                # Remove numeric constraints that are not permitted. (from pydantic ge, le)
                prop.pop("minimum", None)
                prop.pop("maximum", None)

        # Return formatted response format
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        }

    @classmethod
    def _format_response_choice(cls, choice: ChatResponseChoice) -> dict[str, Any]:
        """Format a ChatResponseChoice into OpenAI message format.

        Combines all content items from the choice into a single assistant message.
        This preserves the proper message structure (e.g., tool calls stay with their text).
        """
        return OpenAIMessageConverter.choice_to_provider_message(choice)

    @classmethod
    def convert_message_item(
        cls,
        message_item: MessageItem,
        model_endpoint: AIModelEndpoint | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert a MessageItem to OpenAI message format.

            Handles:
        - Prompt: converts to user/system/assistant message via format_prompt (may return list)
        - ChatResponseChoice: assistant message with all content items (text, tool calls, reasoning, etc.)
        - ToolCallResult: tool message with function output
        - ToolCallResultsMessage: expands grouped tool results into provider messages

            Returns:
                Single dict or list of dicts (Prompt can expand to multiple messages)
        """
        # Case 1: Prompt object (new user/system messages) - may return list
        if isinstance(message_item, Prompt):
            return cls.format_prompt(
                prompt=message_item,
                model_endpoint=model_endpoint,
                **kwargs,
            )

        # Case 2: ToolCallResult (tool execution result)
        if isinstance(message_item, ToolCallResult):
            # OpenAI expects: {"role": "tool", "tool_call_id": "...", "content": "..."}
            return {
                "role": "tool",
                "tool_call_id": message_item.call_id,
                "content": message_item.as_text(),
            }

        # Case 2b: ToolCallResultsMessage (grouped tool execution results)
        if isinstance(message_item, ToolCallResultsMessage):
            return [
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.as_text(),
                }
                for result in message_item.results
            ]

        # Case 3: ChatResponseChoice (assistant response with all content items)
        if isinstance(message_item, ChatResponseChoice):
            return cls._format_response_choice(choice=message_item)

        # Should not reach here due to MessageItem type constraint
        raise ValueError(f"Unsupported message item type: {type(message_item)}")
