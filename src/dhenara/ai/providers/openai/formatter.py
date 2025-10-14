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
from dhenara.ai.types.genai.dhenara.response import (
    ChatResponseChoice,
    ChatResponseReasoningContentItem,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCallContentItem,
)
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

    # ---------------- Responses-specific helpers ----------------
    @classmethod
    def convert_prompt_responses(
        cls,
        formatted_or_provider_prompt: dict,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert a formatted provider prompt dict (as our convert_prompt returns) into Responses input message.

        Accepts provider-formatted dict with keys like {role, content}, where content can be string or list with
        {type: 'text'|'image_url', ...}. Produces Responses message:
        { role, content: [{ type: 'input_text'|'input_image', ... }] }
        """
        role = formatted_or_provider_prompt.get("role")
        content = formatted_or_provider_prompt.get("content")

        # Normalize to a list of items; assistant role expects output_* types per Responses API
        items: list[dict[str, Any]] = []
        is_assistant = role == "assistant"
        text_type = "output_text" if is_assistant else "input_text"
        image_type = "input_image"  # assistant historical images are uncommon; keep as input image for now
        if isinstance(content, list):
            for c in content:
                ctype = c.get("type")
                if ctype == "text":
                    items.append({"type": text_type, "text": c.get("text", "")})
                elif ctype == "image_url":
                    iurl = c.get("image_url", {})
                    url = iurl.get("url")
                    if url:
                        items.append({"type": image_type, "image_url": url})
                else:
                    # Unknown; coerce to text
                    txt = c.get("text") or str(c)
                    items.append({"type": text_type, "text": txt})
        else:
            # Simple string content
            items.append({"type": text_type, "text": str(content) if content is not None else ""})

        return {"role": role, "content": items}

    @classmethod
    def convert_instruction_prompt_responses(
        cls,
        provider_instruction_prompt: dict,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        # provider_instruction_prompt: {role, content: string}
        role = provider_instruction_prompt.get("role", "system")
        content = provider_instruction_prompt.get("content", "")
        # Always input_text for system/user
        return {"role": role, "content": [{"type": "input_text", "text": content}]}

    @classmethod
    def convert_message_item_responses(
        cls,
        message_item: MessageItem,
        model_endpoint: AIModelEndpoint | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Responses equivalent of convert_message_item.

        - Prompt -> convert via format_prompt then convert_prompt_responses
        - ToolCallResult / ToolCallResultsMessage -> map to {role: 'tool', content: [{input_text}]}
        - ChatResponseChoice (assistant prior) -> flatten to {role: 'assistant', content: [{input_text}]}
        """
        if isinstance(message_item, Prompt):
            provider_msg = cls.format_prompt(
                prompt=message_item,
                model_endpoint=model_endpoint,
                **kwargs,
            )
            return cls.convert_prompt_responses(provider_msg, model_endpoint)

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

        if isinstance(message_item, ChatResponseChoice):
            # Convert assistant choice: map all content types to appropriate Responses input items
            # Responses API expects function_call items and message items separately
            import json as _json

            result_items: list[dict] = []

            # Collect text/reasoning for message
            texts: list[str] = []
            for c in message_item.contents:
                if isinstance(c, ChatResponseTextContentItem) and c.text:
                    texts.append(c.text)
                elif isinstance(c, ChatResponseReasoningContentItem) and c.thinking_text:
                    texts.append(c.thinking_text)
                elif isinstance(c, ChatResponseStructuredOutputContentItem) and c.structured_output:
                    try:
                        if c.structured_output.structured_data:
                            texts.append(_json.dumps(c.structured_output.structured_data))
                    except Exception:
                        pass
                elif isinstance(c, ChatResponseToolCallContentItem):
                    # Add function_call items for Responses API
                    tc = c.tool_call
                    result_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": _json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else str(tc.arguments),
                        }
                    )

            # Add message if there's any text content
            if texts:
                result_items.insert(
                    0,
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "\n".join(texts)}],
                    },
                )

            # Return list if multiple items, single dict if one
            if len(result_items) == 1:
                return result_items[0]
            if result_items:
                return result_items
            return {"role": "assistant", "content": [{"type": "output_text", "text": ""}]}

        raise ValueError(f"Unsupported message item type for Responses formatting: {type(message_item)}")

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

    # Responses-specific tool conversion: name is top-level alongside type
    @classmethod
    def convert_tool_responses(
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
    def format_tools_responses(
        cls,
        tools: list[ToolDefinition] | None,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> list[dict] | None:
        if tools:
            return [cls.convert_tool_responses(tool=tool, model_endpoint=model_endpoint) for tool in tools]
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

        - Remove additional keywords from $ref locations (OpenAI doesn't allow description/title with $ref)
        - Remove numeric constraints (minimum/maximum from Pydantic ge/le)
        - Add additionalProperties: false everywhere
        """
        import copy

        schema = copy.deepcopy(schema)

        def clean_object(obj: dict[str, Any]) -> None:
            """Recursively clean a schema object."""
            if isinstance(obj, dict):
                # If this has a $ref, remove all other keys except $ref
                if "$ref" in obj:
                    ref_value = obj["$ref"]
                    obj.clear()
                    obj["$ref"] = ref_value
                    return

                # Set additionalProperties to false for object types
                if obj.get("type") == "object" and "additionalProperties" not in obj:
                    obj["additionalProperties"] = False

                # Remove numeric constraints
                obj.pop("minimum", None)
                obj.pop("maximum", None)
                obj.pop("exclusiveMinimum", None)
                obj.pop("exclusiveMaximum", None)

                # Recursively process nested objects
                for _key, value in list(obj.items()):
                    if isinstance(value, dict):
                        clean_object(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                clean_object(item)

        # Clean root schema
        clean_object(schema)

        # Clean $defs if present
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                clean_object(def_schema)

        return schema

    @classmethod
    def convert_structured_output(
        cls,
        structured_output: StructuredOutputConfig,
        model_endpoint: AIModelEndpoint | None = None,
    ) -> dict[str, Any]:
        """Convert StructuredOutputConfig to OpenAI format"""
        # Get the original JSON schema from Pydantic or dict
        schema = structured_output.get_schema()

        # Clean schema for OpenAI's strict mode
        schema = cls._clean_schema_for_openai_strict_mode(schema)

        # Extract the name from the title and use it for schema name
        schema_name = schema.get("title", "output")

        # Return formatted response format
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                # "strict": True,
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
