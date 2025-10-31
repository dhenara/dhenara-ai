import logging
from typing import Any

from dhenara.ai.providers.base import BaseFormatter
from dhenara.ai.providers.google.message_converter import GoogleMessageConverter
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
from dhenara.ai.types.genai.dhenara.response import ChatResponse
from dhenara.ai.types.shared.file import FileFormatEnum, GenericFile

logger = logging.getLogger(__name__)


class GoogleFormatter(BaseFormatter):
    """
    Formatter for converting Dhenara types to Google-specific formats and vice versa.
    """

    role_map = {
        PromptMessageRoleEnum.USER: "user",
        PromptMessageRoleEnum.ASSISTANT: "model",
        PromptMessageRoleEnum.SYSTEM: "system",  # NOTE:  Don't care as system instructions are taken care separately
    }

    @staticmethod
    def _normalize_tool_response(payload: Any) -> dict[str, Any]:
        """Ensure tool responses satisfy Google's expectation of dict payloads."""

        if isinstance(payload, dict):
            return payload
        return {"result": payload}

    @classmethod
    def convert_prompt(
        cls,
        formatted_prompt: FormattedPrompt,
        model_endpoint: AIModelEndpoint | None = None,
        files: list[GenericFile] | None = None,
        max_words_file: int | None = None,
    ) -> dict[str, Any]:
        # Map Dhenara formats to provider format
        parts = []
        file_contents = None
        if files:
            file_contents = cls.convert_files_to_provider_content(
                files=files,
                model_endpoint=model_endpoint,
                max_words=max_words_file,
            )

        # if model_endpoint.ai_model.functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
        #    return cls._convert_image_model_prompt(
        #        formatted_prompt=formatted_prompt,
        #        model_endpoint=model_endpoint,
        #        file_contents=file_contents,
        #    )

        parts.append(
            {
                "text": formatted_prompt.text,
            }
        )
        if file_contents:
            parts.extend(file_contents)

        role = cls.role_map.get(formatted_prompt.role)
        return {"role": role, "parts": parts}

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

        return {
            "role": role,
            "parts": [
                {
                    "text": formatted_prompt.text,
                }
            ],
        }

    @classmethod
    def convert_files_to_provider_content(
        cls,
        files: list[GenericFile],
        model_endpoint: AIModelEndpoint | None = None,
        max_words: int | None = None,
    ) -> list[dict[str, Any]]:
        # if model_endpoint.ai_model.functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
        #    return cls._convert_files_for_image_models(
        #        files=files,
        #        model_endpoint=model_endpoint,
        #        max_words=max_words,
        #    )

        contents = []
        for file in files:
            file_format = file.get_file_format()
            if file_format in [FileFormatEnum.COMPRESSED, FileFormatEnum.TEXT]:
                text = f"\nFile: {file.get_source_file_name()}  Content: {file.get_processed_file_data(max_words)}"
                pcontent = text
                contents.append(
                    {
                        "text": pcontent,
                    }
                )
            elif file_format in [FileFormatEnum.IMAGE]:
                mime_type = file.get_mime_type()
                contents.append(
                    {
                        "inline_data": {
                            "data": file.get_processed_file_data_content_only(),  # Bytes type
                            "mime_type": mime_type,
                        },
                    }
                )
            else:
                logger.error(f"get_prompt_file_contents: Unknown file_format {file_format} for file {file.name} ")

        return contents

    # -------------------------------------------------------------------------

    # Tools & Structured output
    @classmethod
    def _resolve_refs(cls, schema, root_schema):
        """Recursively resolve $ref pointers in schema"""
        if isinstance(schema, dict):
            # If this is a $ref, replace it with the referenced definition
            if "$ref" in schema:
                ref_path = schema["$ref"]
                # Handle #/$defs/DefinitionName format
                if ref_path.startswith("#/"):
                    parts = ref_path[2:].split("/")
                    ref_target = root_schema
                    for part in parts:
                        ref_target = ref_target.get(part, {})
                    # Return the resolved definition (recursively resolve it too)
                    return cls._resolve_refs(ref_target.copy(), root_schema)
                return schema

            # Recursively resolve refs in all nested values
            resolved = {}
            for key, value in schema.items():
                resolved[key] = cls._resolve_refs(value, root_schema)
            return resolved

        elif isinstance(schema, list):
            return [cls._resolve_refs(item, root_schema) for item in schema]

        return schema

    @classmethod
    def _clean_schema_for_google(cls, schema):
        """Remove Google-incompatible fields from schema"""
        if isinstance(schema, dict):
            # Remove additionalProperties: false
            if "additionalProperties" in schema and schema["additionalProperties"] is False:
                del schema["additionalProperties"]

            # Remove empty examples lists
            if "examples" in schema and not schema["examples"]:
                del schema["examples"]

            # Remove $defs since all refs have been resolved
            if "$defs" in schema:
                del schema["$defs"]

            # Process nested schemas
            for _key, value in list(schema.items()):
                if isinstance(value, (dict, list)):
                    cls._clean_schema_for_google(value)

        elif isinstance(schema, list):
            for item in schema:
                if isinstance(item, (dict, list)):
                    cls._clean_schema_for_google(item)

        return schema

    @classmethod
    def convert_function_parameter(
        cls,
        param: FunctionParameter,
        root_schema: dict | None = None,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionParameter to Google format (simple JSON Schema).

        Keep this minimal: only emit standard JSON Schema fields the Gemini SDK expects.
        We inline complex shapes upstream (runner) so no $ref resolution here.
        """
        result: dict[str, Any] = {
            "type": param.type,
        }
        if getattr(param, "description", None):
            result["description"] = param.description
        # Array item schema (already inlined by runner if it originated from a $ref)
        if getattr(param, "items", None):
            result["items"] = param.items
        # Enumerations (optional)
        if getattr(param, "allowed_values", None):
            result["enum"] = param.allowed_values
        # Default (optional)
        if getattr(param, "default", None) is not None:
            result["default"] = param.default

        return result

    @classmethod
    def convert_function_parameters(
        cls,
        params: FunctionParameters,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionParameters to Google format (simple JSON Schema)."""
        result = {
            "type": "object",
            "properties": {name: cls.convert_function_parameter(param) for name, param in params.properties.items()},
        }
        # Include required only when non-empty
        required_params = [name for name, param in params.properties.items() if param.required]
        if required_params:
            result["required"] = required_params
        elif params.required:
            # Fallback to any explicitly provided list
            if params.required:
                result["required"] = list(params.required)

        return result

    @classmethod
    def convert_function_definition(
        cls,
        func_def: FunctionDefinition,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionDefinition to Google format"""
        return {
            "name": func_def.name,
            "description": func_def.description,
            "parameters": cls.convert_function_parameters(func_def.parameters),
        }

    @classmethod
    def convert_tool(
        cls,
        tool: ToolDefinition,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> Any:
        """Convert ToolDefinition to Google format"""

        return {"function_declarations": [cls.convert_function_definition(tool.function)]}

    @classmethod
    def convert_tool_choice(
        cls,
        tool_choice: ToolChoice,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> Any:
        """Convert ToolChoice to Google format"""

        if tool_choice.type is None:
            return None
        elif tool_choice.type == "zero_or_more":
            _cfg = {
                "mode": "AUTO",
            }
        elif tool_choice.type == "one_or_more":
            _cfg = {
                "mode": "ANY",
            }
        elif tool_choice.type == "specific":
            _cfg = {
                "mode": "AUTO",
                "allowed_function_names": [tool_choice.specific_tool_name],
            }

        return {"function_calling_config": _cfg}

    @classmethod
    def convert_structured_output(
        cls,
        structured_output: StructuredOutputConfig,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert StructuredOutputConfig to Google format"""
        # Get the original JSON schema from Pydantic or dict
        _schema = structured_output.get_schema()
        # First resolve all $ref pointers
        _schema = cls._resolve_refs(_schema, _schema)
        # Then clean up Google-incompatible fields
        return cls._clean_schema_for_google(_schema)

    @classmethod
    def convert_dai_message_item_to_provider(
        cls,
        message_item: MessageItem,
        model_endpoint: AIModelEndpoint | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert a MessageItem to Google/Gemini message format.

            Handles:
        - Prompt: converts to user/model message via format_prompt (may return list)
        - ChatResponseChoice: model message with all content items (text, function_call parts, etc.)
            Delegates to GoogleMessageConverter.dai_choice_to_provider_message.
        - ToolCallResult: user message with function_response part
        - ToolCallResultsMessage: user message with multiple function_response parts

            Returns:
                Single dict or list of dicts (Prompt can expand to multiple messages)
        """
        # Case 1: Prompt object (new user/model messages) - may return list
        if isinstance(message_item, Prompt):
            return cls.format_prompt(
                prompt=message_item,
                model_endpoint=model_endpoint,
                **kwargs,
            )

        # Case 2: ToolCallResult (tool execution result)
        if isinstance(message_item, ToolCallResult):
            # Google/Gemini expects function responses in user role with function_response part:
            # {"role": "user", "parts": [{"function_response": {"name": "...", "response": {...}}}]}
            response_payload = cls._normalize_tool_response(message_item.as_json())
            return {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": message_item.name or "unknown_function",
                            "response": response_payload,
                        }
                    }
                ],
            }

        # Case 2b: ToolCallResultsMessage (grouped tool execution results)
        if isinstance(message_item, ToolCallResultsMessage):
            return {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": result.name or "unknown_function",
                            "response": cls._normalize_tool_response(result.as_json()),
                        }
                    }
                    for result in message_item.results
                ],
            }

        # Case 3: ChatResponse (model response with all content items)
        # Delegate to message converter (single source of truth for ChatResponse conversions)
        if isinstance(message_item, ChatResponse):
            return GoogleMessageConverter.dai_response_to_provider_message(
                dai_response=message_item,
                model_endpoint=model_endpoint,
            )

        # Should not reach here due to MessageItem type constraint
        raise ValueError(f"Unsupported message item type: {type(message_item)}")
