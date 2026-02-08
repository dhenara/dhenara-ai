import logging
from typing import Any

from dhenara.ai.providers.anthropic.message_converter import AnthropicMessageConverter
from dhenara.ai.providers.base import BaseFormatter
from dhenara.ai.types.genai.ai_model import AIModelEndpoint
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
from dhenara.ai.types.shared.file import FileFormatEnum, GenericFile, ProcessedFile

logger = logging.getLogger(__name__)


class AnthropicFormatter(BaseFormatter):
    """
    Formatter for converting Dhenara types to Anthropic-specific formats and vice versa.
    """

    role_map = {
        PromptMessageRoleEnum.USER: "user",
        PromptMessageRoleEnum.ASSISTANT: "assistant",
        PromptMessageRoleEnum.SYSTEM: "system",  # NOTE: Don't care as system instructions are taken care separately
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
        # There is no native support for `system` prompt. So set role always send as `user` role,
        # so that beta models can send them as prompt.
        # For other models, the text will be send as seperate argument
        role = cls.role_map.get(PromptMessageRoleEnum.USER)

        return {"role": role, "content": formatted_prompt.text}

    @classmethod
    def convert_files_to_provider_content(
        cls,
        files: list[GenericFile],
        model_endpoint: AIModelEndpoint | None = None,
        max_words: int | None = None,
    ) -> list[dict[str, Any]]:
        contents = []

        for file in files:
            if not isinstance(file, ProcessedFile):
                logger.error(f"convert_files_to_provider_content: expected ProcessedFile, got {type(file)}")
                continue

            file_format = file.get_file_format()
            try:
                if file_format in [FileFormatEnum.COMPRESSED, FileFormatEnum.TEXT]:
                    text = (
                        f"\nFile: {file.get_source_file_name()}  "
                        f"Content: {file.get_processed_file_data(max_words=max_words)}"
                    )
                    contents.append(
                        {
                            "type": "text",
                            "text": text,
                        }
                    )
                elif file_format == FileFormatEnum.IMAGE:
                    mime_type = file.get_mime_type()
                    if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                        raise ValueError(f"Unsupported media type: {mime_type}")

                    contents.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": file.get_processed_file_data_content_only(),
                            },
                        }
                    )
                else:
                    logger.error(f"Unknown file_format {file_format} for file {file.name}")
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {e}")

        return contents

    # -------------------------------------------------------------------------
    # Tools & Structured output
    @classmethod
    def convert_function_parameter(
        cls,
        param: FunctionParameter,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        result = param.model_dump(
            exclude={"required", "allowed_values", "default"},
            exclude_none=True,  # Exclude None values to avoid invalid schema fields
        )
        if param.allowed_values is not None:
            result["enum"] = param.allowed_values

        return result

    @classmethod
    def convert_function_parameters(
        cls,
        params: FunctionParameters,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionParameters to Anthropic format"""
        # Create a new dictionary with transformed properties
        result: dict[str, Any] = {
            "type": params.type,
            "properties": {name: cls.convert_function_parameter(param) for name, param in params.properties.items()},
        }

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
        """Convert to Anthropic format"""
        return {
            "name": func_def.name,
            "description": func_def.description,
            "input_schema": cls.convert_function_parameters(func_def.parameters),
        }

    @classmethod
    def convert_tool(
        cls,
        tool: ToolDefinition,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> Any:
        """Convert ToolDefinition to Anthropic format"""
        return cls.convert_function_definition(tool.function)

    @classmethod
    def convert_tool_choice(
        cls,
        tool_choice: ToolChoice,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> Any:
        """Convert ToolChoice to Anthropic format"""

        if tool_choice.type is None:
            return None
        elif tool_choice.type == "zero_or_more":
            return {"type": "auto"}
        elif tool_choice.type == "one_or_more":
            return {"type": "any"}
        elif tool_choice.type == "specific":
            return {"type": "tool", "name": tool_choice.specific_tool_name}

    @classmethod
    def convert_structured_output(
        cls,
        structured_output: StructuredOutputConfig,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """
        Convert structured output config to Anthropic tool format.
        Since Anthropic doesn't directly support structured output,
        we create a specialized tool and force the model to use it.
        """
        schema = structured_output.get_schema()
        name = schema.pop("title", None) or "structured_output"
        description = "Generate structured output according to schema"

        # Clean up schema for Anthropic compatibility
        if "properties" in schema:
            for prop in schema["properties"].values():
                # Remove unsupported validations
                for field in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
                    if field in prop:
                        prop.pop(field)

                # Convert enum to Anthropic format
                if "enum" in prop:
                    prop["enum"] = list(prop["enum"])

        # Create the tool
        tool = {
            "name": name,
            "description": description,
            "input_schema": schema,
        }

        return tool

    @classmethod
    def _transform_schema_for_native_structured_output(
        cls,
        structured_output: StructuredOutputConfig,
    ) -> dict[str, Any]:
        """Return an Anthropic-compatible JSON schema for `output_config.format`.

        Keep this aligned with OpenAI/Google patterns: start from the Pydantic JSON schema
        (via StructuredOutputConfig.get_schema()) and apply a lightweight compatibility transform.
        """
        schema = structured_output.get_schema()
        if not schema:
            return {"type": "object"}

        # Manual transform: remove common unsupported constraints and force additionalProperties: false.
        # This is intentionally conservative; Anthropic SDK transform_schema() is preferred.
        unsupported_keys = {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
            "pattern",
        }

        def _walk(node: Any) -> Any:
            if isinstance(node, list):
                return [_walk(x) for x in node]
            if not isinstance(node, dict):
                return node

            out: dict[str, Any] = {}
            for k, v in node.items():
                if k in unsupported_keys:
                    continue
                out[k] = _walk(v)

            if out.get("type") == "object":
                out.setdefault("additionalProperties", False)

            # Normalize enum to list
            if "enum" in out and isinstance(out["enum"], (set, tuple)):
                out["enum"] = list(out["enum"])

            return out

        return _walk(schema)

    @classmethod
    def format_structured_output_output_config(
        cls,
        structured_output: StructuredOutputConfig,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Build Anthropic `output_config` payload for native JSON-schema outputs."""
        schema = cls._transform_schema_for_native_structured_output(structured_output)
        return {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }

    @classmethod
    def convert_dai_message_item_to_provider(
        cls,
        message_item: MessageItem,
        model_endpoint: AIModelEndpoint | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert a MessageItem to Anthropic message format.

            Handles:
        - Prompt: converts to user/assistant message via format_prompt (may return list)
        - ChatResponseChoice: assistant message with all content items (text, tool_use blocks, etc.)
            Delegates to AnthropicMessageConverter.dai_choice_to_provider_message.
        - ToolCallResult: user message with tool_result content block
        - ToolCallResultsMessage: user message containing multiple tool_result blocks

            Returns:
                Single dict or list of dicts (Prompt can expand to multiple messages)
        """
        # Case 1: Prompt object (new user/assistant messages) - may return list
        if isinstance(message_item, Prompt):
            res = cls.format_prompt(
                prompt=message_item,
                model_endpoint=model_endpoint,
                **kwargs,
            )
            if isinstance(res, str):
                raise ValueError(
                    "AnthropicFormatter: Prompt formatting returned a string; expected dict provider message"
                )
            return res

        # Case 2: ToolCallResult (tool execution result)
        if isinstance(message_item, ToolCallResult):
            # Anthropic expects tool results in user messages:
            # {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message_item.call_id,
                        "content": message_item.as_text(),
                    }
                ],
            }

        # Case 2b: ToolCallResultsMessage (grouped tool execution results)
        if isinstance(message_item, ToolCallResultsMessage):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.as_text(),
                    }
                    for result in message_item.results
                ],
            }

        # Case 3: ChatResponse (assistant response with all content items)
        # Delegate to message converter (single source of truth for ChatResponse conversions)
        return AnthropicMessageConverter.dai_response_to_provider_message(
            dai_response=message_item,
            model_endpoint=model_endpoint,
        )
