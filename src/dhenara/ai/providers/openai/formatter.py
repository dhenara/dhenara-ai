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
from dhenara.ai.types.genai.dhenara.response import ChatResponse
from dhenara.ai.types.shared.file import FileFormatEnum, GenericFile

logger = logging.getLogger(__name__)


class OpenAIFormatter(BaseFormatter):
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
            if isinstance(formatted_prompt, FormattedPrompt):
                # Use Simple text format
                content = formatted_prompt.text
            else:
                raise ValueError(
                    "Prompt must be of type FormattedPrompt, provided:"
                    f"{type(formatted_prompt)} , value: {formatted_prompt}"
                )

        role = cls.role_map.get(formatted_prompt.role)
        return {"role": role, "content": content}

    @classmethod
    def convert_instruction_prompt(
        cls,
        formatted_prompt: FormattedPrompt,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        # For Responses API, instructions are just text
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
        """Convert ToolDefinition to OpenAI Responses format.

        In Responses API, tools expect function name at the top-level next to type.
        Example:
        {"type": "function", "name": "foo", "parameters": {...}, "description": "..."}
        """
        func_def = tool.function
        res: dict[str, Any] = {
            "type": "function",
            "name": func_def.name,
            "parameters": cls.convert_function_parameters(func_def.parameters),
        }
        if getattr(func_def, "description", None):
            res["description"] = func_def.description
        return res

    @classmethod
    def format_tools(
        cls,
        tools: list[ToolDefinition] | None,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> list[dict] | None:
        if tools:
            return [cls.convert_tool(tool=tool, model_endpoint=model_endpoint) for tool in tools]
        return None

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
    def _clean_schema_for_openai_strict_mode(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Clean JSON schema to comply with OpenAI's strict mode requirements.

        - For any object types, ensure additionalProperties: false
        - In branches (anyOf/oneOf/allOf), recursively enforce the same
        - For array types, clean the items schema
        - If a dict contains $ref, strip other keys alongside it
        - Remove unsupported/min/max style numeric constraints to reduce friction
        """
        import copy

        schema = copy.deepcopy(schema)

        def clean_object(obj: dict[str, Any]) -> None:
            if not isinstance(obj, dict):
                return

            # If this has a $ref, keep only $ref
            if "$ref" in obj:
                ref_value = obj["$ref"]
                obj.clear()
                obj["$ref"] = ref_value
                return

            # Normalize object schemas
            if obj.get("type") == "object":
                # Force strictness: OpenAI requires additionalProperties to be false for all objects
                obj["additionalProperties"] = False
                # Ensure properties exists
                if not isinstance(obj.get("properties"), dict):
                    obj["properties"] = {}
                # OpenAI strict mode requires 'required' to list every key in 'properties'
                prop_keys = list(obj["properties"].keys())
                existing_required = obj.get("required")
                if not isinstance(existing_required, list) or set(existing_required) != set(prop_keys):
                    obj["required"] = prop_keys

            # Remove numeric constraints that sometimes appear from pydantic
            for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                if k in obj:
                    obj.pop(k, None)

            # Recurse into common schema containers
            for branch in ("properties", "patternProperties"):
                if isinstance(obj.get(branch), dict):
                    for _k, v in list(obj[branch].items()):
                        if isinstance(v, dict):
                            clean_object(v)

            for branch in ("anyOf", "oneOf", "allOf"):
                if isinstance(obj.get(branch), list):
                    for item in obj[branch]:
                        if isinstance(item, dict):
                            clean_object(item)

            # Arrays: clean items
            if obj.get("type") == "array" and isinstance(obj.get("items"), dict):
                clean_object(obj["items"])

        # Clean root schema
        clean_object(schema)

        # Clean $defs if present
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                if isinstance(def_schema, dict):
                    clean_object(def_schema)

        return schema

    @classmethod
    def convert_structured_output(
        cls,
        structured_output: StructuredOutputConfig,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert StructuredOutputConfig to OpenAI format"""
        # Get the original JSON schema from Pydantic or dict and clean for strict mode
        schema = structured_output.get_schema()
        schema = cls._clean_schema_for_openai_strict_mode(schema)

        # Extract the name from the title and use it for schema name
        schema_name = schema.get("title", "output")

        # Return formatted response format
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        }

    # -------------------------------------------------------------------------
    # Dhenara <-> OpenAI conversions
    @classmethod
    def convert_dai_message_item_to_provider(
        cls,
        message_item: MessageItem,
        model_endpoint: AIModelEndpoint | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Responses equivalent of convert_dai_message_item_to_provider.

        - Prompt -> convert via format_prompt then convert_prompt_responses
        - ToolCallResult / ToolCallResultsMessage -> map to {role: 'tool', content: [{input_text}]}
        - ChatResponseChoice (assistant prior) -> flatten to {role: 'assistant', content: [{input_text}]}
        """
        if isinstance(message_item, Prompt):
            return cls.format_prompt(
                prompt=message_item,
                model_endpoint=model_endpoint,
                **kwargs,
            )

        if isinstance(message_item, ToolCallResult):
            # Responses API uses function_call_output, not role='tool'
            return {
                "type": "function_call_output",
                "call_id": message_item.call_id,
                "output": message_item.as_text(),
            }

        if isinstance(message_item, ToolCallResultsMessage):
            # Return list of function_call_output items
            return [
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.as_text(),
                }
                for result in message_item.results
            ]

        if isinstance(message_item, ChatResponse):
            return OpenAIMessageConverter.dai_response_to_provider_message(
                dai_response=message_item,
                model_endpoint=model_endpoint,
            )

        raise ValueError(f"Unsupported message item type for Responses formatting: {type(message_item)}")
