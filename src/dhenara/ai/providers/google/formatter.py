import logging
from typing import Any

from dhenara.ai.providers.base import BaseFormatter
from dhenara.ai.types.genai.ai_model import AIModelEndpoint
from dhenara.ai.types.genai.dhenara.request import (
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    PromptMessageRoleEnum,
    StructuredOutputConfig,
    ToolChoice,
    ToolDefinition,
)
from dhenara.ai.types.genai.dhenara.request.data import FormattedPrompt
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
    def convert_function_parameter(
        cls,
        param: FunctionParameter,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionParameter to Google format"""
        result = param.model_dump(
            exclude={"required", "allowed_values", "default"},
        )
        return result

    @classmethod
    def convert_function_parameters(
        cls,
        params: FunctionParameters,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert FunctionParameters to Google format"""
        # Create a new dictionary with transformed properties
        result = {
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
        # return structured_output.output_schema

        def _clean_schema_for_api(schema):
            """Remove `additionalProperties` in nested list/dict"""
            if isinstance(schema, dict):
                # Remove additionalProperties: false
                if "additionalProperties" in schema and schema["additionalProperties"] is False:
                    del schema["additionalProperties"]

                # Remove empty examples lists
                if "examples" in schema and not schema["examples"]:
                    del schema["examples"]

                # Process nested schemas
                for _key, value in list(schema.items()):
                    if isinstance(value, (dict, list)):
                        _clean_schema_for_api(value)

            elif isinstance(schema, list):
                for item in schema:
                    if isinstance(item, (dict, list)):
                        _clean_schema_for_api(item)

            return schema

        # Get the original JSON schema from Pydantic or dict
        _schema = structured_output.get_schema()
        return _clean_schema_for_api(_schema)
