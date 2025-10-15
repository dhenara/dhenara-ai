import json
import logging
from typing import Any

from dhenara.ai.providers.openai import OpenAIClientBase
from dhenara.ai.providers.openai.formatter import OpenAIFormatter
from dhenara.ai.providers.openai.message_converter import OpenAIMessageConverter
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseContentItem,
    ChatResponseReasoningContentItemDelta,
    ChatResponseTextContentItemDelta,
    ChatResponseUsage,
    StreamingChatResponse,
)
from dhenara.ai.types.genai.ai_model import AIModelAPIProviderEnum
from dhenara.ai.types.shared.api import SSEErrorResponse

logger = logging.getLogger(__name__)


class OpenAIResponses(OpenAIClientBase):
    """OpenAI Responses API client (text + tools + structured output).

    Phase 1 scope:
    - OpenAI provider only (not Azure variants)
    - Text and tools, structured output, reasoning, streaming events (text deltas)
    - Vision inputs supported via input_image (data URL) when files are provided
    - Image generation remains in image.py (legacy endpoint)
    """

    # ----------------------- Request build -----------------------
    def _build_responses_input(
        self,
        *,
        prompt: dict | None,
        context: list[dict] | None,
        instructions: dict | None,
        messages: list | None,
    ) -> list[dict]:
        """Build the Responses API 'input' array of role/content items."""
        input_items: list[dict] = []

        # Instructions as a system message if provided
        if instructions:
            input_items.append(OpenAIFormatter.convert_instruction_prompt_responses(instructions))

        if messages is not None:
            # Convert Dhenara messages to Responses input messages
            for mi in messages:
                converted = OpenAIFormatter.convert_message_item_responses(
                    message_item=mi,
                    model_endpoint=self.model_endpoint,
                )
                if isinstance(converted, list):
                    input_items.extend(converted)
                elif converted:
                    input_items.append(converted)
        else:
            # Context (system/user messages before prompt)
            if context:
                input_items.extend([OpenAIFormatter.convert_prompt_responses(c, self.model_endpoint) for c in context])

            if prompt is not None:
                input_items.append(OpenAIFormatter.convert_prompt_responses(prompt, self.model_endpoint))

        return input_items

    def get_api_call_params(
        self,
        prompt: dict | None,
        context: list[dict] | None = None,
        instructions: dict | None = None,
        messages: list | None = None,
    ) -> AIModelCallResponse:
        if not self._client:
            raise RuntimeError("Client not initialized. Use with 'async with' context manager")

        if self._input_validation_pending:
            raise ValueError("inputs must be validated with `self.validate_inputs()` before api calls")

        api = self.model_endpoint.api
        if api.provider != AIModelAPIProviderEnum.OPEN_AI:
            raise ValueError("OpenAIResponses only supports AIModelAPIProviderEnum.OPEN_AI in Phase 1")

        input_messages = self._build_responses_input(
            prompt=prompt,
            context=context or [],
            instructions=instructions,
            messages=messages,
        )

        # Base args
        args: dict[str, Any] = {
            "model": self.model_name_in_api_calls,
            "input": input_messages,
            "stream": self.config.streaming,
        }

        # Max tokens (Responses uses max_output_tokens)
        # Note: Responses API doesn't have a separate max_reasoning_tokens parameter.
        # Reasoning tokens come out of max_output_tokens budget.
        max_output_tokens, max_reasoning_tokens = self.config.get_max_output_tokens(self.model_endpoint.ai_model)
        if max_output_tokens is not None:
            args["max_output_tokens"] = max_output_tokens

        # Reasoning configuration
        # Responses API: reasoning = {effort: "low"|"medium"|"high",
        #                             generate_summary: "auto"|"concise"|"detailed"}
        # Note: Unlike Anthropic's budget_tokens, OpenAI uses max_output_tokens for
        # total (text + reasoning)
        if self.config.reasoning:
            reasoning_config: dict[str, Any] = {}

            # Effort level
            if self.config.reasoning_effort is not None:
                effort = self.config.reasoning_effort
                # Normalize Dhenara "minimal" -> OpenAI "low"
                if isinstance(effort, str) and effort.lower() == "minimal":
                    effort = "low"
                reasoning_config["effort"] = effort
            else:
                # Default effort if reasoning is enabled but not specified
                reasoning_config["effort"] = "low"

            # Inorder to get the reasoning text, OpenAI need to pass `summary` as any of the
            # : "auto", "concise", "detailed"
            reasoning_config["summary"] = "auto"

            if reasoning_config:
                args["reasoning"] = reasoning_config

        # Log warning about reasoning token budget (informational)
        if max_reasoning_tokens is not None:
            logger.debug(
                f"Responses API: max_reasoning_tokens ({max_reasoning_tokens}) is advisory only. "
                f"Reasoning tokens come from max_output_tokens budget ({max_output_tokens})."
            )

        # Tools and tool choice
        if self.config.tools:
            # Use Responses-specific tool schema (top-level name)
            tools_formatted = None
            try:
                tools_formatted = self.formatter.format_tools_responses(
                    tools=self.config.tools,
                    model_endpoint=self.model_endpoint,
                )
            except Exception:
                # Fallback to generic formatter if Responses variant not available
                tools_formatted = self.formatter.format_tools(
                    tools=self.config.tools,
                    model_endpoint=self.model_endpoint,
                )
            if tools_formatted:
                args["tools"] = tools_formatted
        if self.config.tool_choice:
            args["tool_choice"] = self.formatter.format_tool_choice(
                tool_choice=self.config.tool_choice,
                model_endpoint=self.model_endpoint,
            )

        # Structured output via text.format parameter
        if self.config.structured_output:
            # Responses API uses text.format for structured output, not response_format
            schema_dict = self.formatter.convert_structured_output(
                structured_output=self.config.structured_output,
                model_endpoint=self.model_endpoint,
            )
            # Extract json_schema from the Chat-style format
            if schema_dict.get("type") == "json_schema" and "json_schema" in schema_dict:
                json_schema_info = schema_dict["json_schema"]
                args["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema_info.get("name", "output"),
                        "schema": json_schema_info.get("schema", {}),
                        "strict": json_schema_info.get("strict", True),
                    }
                }
                if "description" in json_schema_info:
                    args["text"]["format"]["description"] = json_schema_info["description"]

        # Metadata: attach user id if available
        user = self.config.get_user()
        if user:
            args["metadata"] = {"user_id": user}

        # Streaming options
        # Note: Some SDK versions of Responses API do not support stream_options.include_usage.
        # We'll omit stream_options to avoid 400 errors and rely on usage in the final response.
        # if self.config.streaming:
        #     args["stream_options"] = {"include_usage": True}

        # Extra options passthrough (allow overrides)
        if self.config.options:
            args.update(self.config.options)

        return {"response_args": args}

    # ----------------------- API calls -----------------------

    def do_api_call_sync(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        args = dict(api_call_params["response_args"])
        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError("OpenAIResponses doens't supports AIModelAPIProviderEnum.MICROSOFT_AZURE_AI in Phase 1")
        else:
            response = self._client.responses.create(**args)
        return response

    async def do_api_call_async(
        self,
        api_call_params: dict,
    ) -> AIModelCallResponse:
        args = dict(api_call_params["response_args"])
        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError("OpenAIResponses doens't supports AIModelAPIProviderEnum.MICROSOFT_AZURE_AI in Phase 1")
        else:
            response = await self._client.responses.create(**args)
        return response

    def do_streaming_api_call_sync(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        args = dict(api_call_params["response_args"])
        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError("OpenAIResponses doens't supports AIModelAPIProviderEnum.MICROSOFT_AZURE_AI in Phase 1")
        else:
            stream = self._client.responses.create(**args)

        return stream

    async def do_streaming_api_call_async(
        self,
        api_call_params,
    ) -> AIModelCallResponse:
        args = dict(api_call_params["response_args"])
        if self.model_endpoint.api.provider == AIModelAPIProviderEnum.MICROSOFT_AZURE_AI:
            raise ValueError("OpenAIResponses doens't supports AIModelAPIProviderEnum.MICROSOFT_AZURE_AI in Phase 1")
        else:
            stream = await self._client.responses.create(**args)

        return stream

    # ----------------------- Parsing -----------------------
    def _get_usage_from_provider_response(self, response) -> ChatResponseUsage | None:
        try:
            usage = getattr(response, "usage", None)
            if not usage:
                return None
            # Responses usage typically has input_tokens and output_tokens
            total = getattr(usage, "total_tokens", None)
            prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)

            # Extract reasoning tokens from output_tokens_details
            reasoning_tokens = None
            output_details = getattr(usage, "output_tokens_details", None)
            if output_details:
                reasoning_tokens = getattr(output_details, "reasoning_tokens", None)

            if total is None and prompt is not None and completion is not None:
                total = int(prompt) + int(completion)

            return ChatResponseUsage(
                total_tokens=total,
                prompt_tokens=prompt,
                completion_tokens=completion,
                reasoning_tokens=reasoning_tokens,
            )
        except Exception as e:
            logger.debug(f"_get_usage_from_provider_response (Responses): {e}")
            return None

    def _parse_tool_arguments(self, arguments: str | dict) -> dict:
        """Parse tool call arguments from string or dict."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except Exception:
                logger.debug(f"Failed to parse tool arguments as JSON: {arguments}")
                return {}
        return {}

    def parse_response(self, response) -> ChatResponse:
        """Parse OpenAI Responses API response into Dhenara ChatResponse.

        Response structure:
        - response.output: list of output items (message, reasoning, function_call, etc.)
        - response.reasoning: reasoning config/summary (root level)
        - response.output_text: convenience field (text from first message)

        Output item types:
        - ResponseOutputMessage (type='message'): contains content list with ResponseOutputText items
        - ResponseReasoningItem (type='reasoning'): thinking/reasoning block
        - ResponseFunctionToolCall (type='function_call'): tool call with name/arguments/call_id
        """
        model = getattr(response, "model", None) or self.model_endpoint.ai_model.model_name
        contents: list[ChatResponseContentItem] = []
        content_index = 0

        # Parse output array for all content types
        output = getattr(response, "output", None) or []

        # Check for incomplete response (reasoning models may use all tokens for thinking)
        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        if status == "incomplete" and incomplete_details:
            reason = getattr(incomplete_details, "reason", None)
            logger.warning(f"Incomplete response: reason={reason}")

        for item in output:
            converted = OpenAIMessageConverter.provider_message_to_content_items_responses_api(
                output_item=item,
                role="assistant",
                index_start=content_index,
                ai_model_provider=self.model_endpoint.ai_model.provider,
                structured_output_config=self.config.structured_output,
            )
            if not converted:
                continue
            for ci in converted:
                # Normalize incremental indices
                ci.index = content_index
                contents.append(ci)
                content_index += 1

        usage, usage_charge = self.get_usage_and_charge(response)

        choice = ChatResponseChoice(
            index=0,
            finish_reason=None,
            stop_sequence=None,
            contents=contents,
            metadata={},
        )

        return ChatResponse(
            model=model,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            usage=usage,
            usage_charge=usage_charge,
            choices=[choice],
            metadata=AIModelCallResponseMetaData(
                streaming=False,
                duration_seconds=0,
                provider_metadata={
                    "id": getattr(response, "id", None),
                    "created": str(getattr(response, "created", "")),
                    "object": getattr(response, "object", None),
                },
            ),
        )

    # Streaming handlers: convert Responses events to StreamingChatResponse
    def parse_stream_chunk(self, chunk) -> StreamingChatResponse | SSEErrorResponse | None:
        """Parse Responses API streaming chunks.

        Streaming event types mirror the output array structure:
        - response.output_text.delta: text content delta
        - response.reasoning.delta: reasoning/thinking block delta
        - response.function_call.delta: tool call delta (name/arguments)
        - response.done: final usage/metadata
        """
        processed: list[StreamingChatResponse] = []

        # Provider metadata (grab once)
        if not self.streaming_manager.provider_metadata:
            self.streaming_manager.provider_metadata = {
                "id": getattr(chunk, "id", None),
                "created": str(getattr(chunk, "created", "")),
                "object": getattr(chunk, "object", None),
                "system_fingerprint": getattr(chunk, "system_fingerprint", None),
            }

        # Handle usage if present (typically in 'done' or 'completed' event)
        # Check both chunk.usage and chunk.response.usage
        usage = getattr(chunk, "usage", None)
        if not usage:
            # For events like response.completed, usage is in chunk.response.usage
            response_obj = getattr(chunk, "response", None)
            if response_obj:
                usage = getattr(response_obj, "usage", None)

        if usage:
            try:
                # Extract reasoning tokens from output_tokens_details
                reasoning_tokens = None
                output_details = getattr(usage, "output_tokens_details", None)
                if output_details:
                    reasoning_tokens = getattr(output_details, "reasoning_tokens", None)

                usage_obj = ChatResponseUsage(
                    total_tokens=getattr(usage, "total_tokens", None),
                    prompt_tokens=getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", None),
                    reasoning_tokens=reasoning_tokens,
                )
                self.streaming_manager.update_usage(usage_obj)
            except Exception:
                pass

        # Event type dispatch
        event_type = getattr(chunk, "type", None)
        if not event_type:
            logger.debug(f"parse_stream_chunk: no event_type, chunk={chunk}")
            return processed

        logger.debug(f"parse_stream_chunk: event_type={event_type}, chunk_type={type(chunk)}")

        # Text delta (message output_text)
        if event_type == "response.output_text.delta":
            delta_text = getattr(chunk, "delta", None)
            if delta_text:
                content_delta = ChatResponseTextContentItemDelta(
                    index=0,
                    role="assistant",
                    text_delta=delta_text,
                    metadata={},
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        # Reasoning delta (from reasoning models like o3-mini)
        # Note: reasoning has its own dedicated text stream separate from reasoning_summary
        elif event_type == "response.reasoning_text.delta":
            delta_text = getattr(chunk, "delta", None)
            if delta_text:
                content_delta = ChatResponseReasoningContentItemDelta(
                    index=0,
                    role="assistant",
                    thinking_text_delta=delta_text,
                    # TODO: take care of summary, id and signature
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        # Reasoning summary delta (condensed reasoning for o3-mini)
        elif event_type == "response.reasoning_summary_text.delta":
            delta_text = getattr(chunk, "delta", None)
            if delta_text:
                content_delta = ChatResponseReasoningContentItemDelta(
                    index=0,
                    role="assistant",
                    thinking_text_delta=delta_text,
                )
                choice_delta = ChatResponseChoiceDelta(
                    index=0,
                    finish_reason=None,
                    stop_sequence=None,
                    content_deltas=[content_delta],
                    metadata={},
                )
                response_chunk = self.streaming_manager.update(choice_deltas=[choice_delta])
                processed.append(StreamingChatResponse(id=getattr(chunk, "id", None), data=response_chunk))

        # Function call arguments delta (tool calls)
        elif event_type == "response.function_call_arguments.delta":
            # Tool calls stream incrementally; we accumulate them
            # The streaming manager should handle partial tool call assembly
            # For now, we'll skip tool call streaming deltas as they're complex
            # and require state tracking across multiple chunks
            pass

        # Completion event - usage already handled above
        elif event_type == "response.completed":
            # Mark as done
            pass

        # Error events
        elif event_type == "error":
            error_obj = getattr(chunk, "error", None)
            if error_obj:
                logger.error(f"Streaming error: {error_obj}")

        return processed
