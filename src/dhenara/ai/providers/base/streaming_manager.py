import logging
from datetime import datetime as datetime_type

from dhenara.ai.config import settings
from dhenara.ai.types.genai import (
    AIModelCallResponse,
    AIModelCallResponseMetaData,
    AIModelEndpoint,
    AIModelFunctionalTypeEnum,
    ChatMessageContentPart,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseChunk,
    ChatResponseContentItemType,
    ChatResponseGenericContentItem,
    ChatResponseReasoningContentItem,
    ChatResponseStructuredOutput,
    ChatResponseStructuredOutputContentItem,
    ChatResponseTextContentItem,
    ChatResponseToolCall,
    ChatResponseToolCallContentItem,
    ChatResponseUsage,
    ExternalApiCallStatus,
    ExternalApiCallStatusEnum,
    ImageResponseUsage,
    StreamingChatResponse,
    UsageCharge,
)
from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.shared.base import BaseModel

logger = logging.getLogger(__name__)


class INTStreamingProgress(BaseModel):
    """INTERNAL : Tracks the progress of a streaming response"""

    # total_content: str = ""
    updates_count: int = 0
    start_time: datetime_type
    last_token_time: datetime_type
    is_complete: bool = False
    # Add tracking for Deepseek thinking state, which is embedded in content
    in_thinking_block: bool = False


class StreamingManager:
    """Manages streaming state and constructs final ChatResponse"""

    def __init__(
        self,
        model_endpoint: AIModelEndpoint,
        structured_output_config: StructuredOutputConfig | None = None,
    ):
        self.model_endpoint = model_endpoint
        self.structured_output_config = structured_output_config

        # Fields required  to create  final ChatResponse
        # self.final_response: ChatResponse | None = None
        self.usage: ChatResponseUsage | None = None
        self.usage_charge: UsageCharge | None = None
        self.choices: list[ChatResponseChoice] = []
        self.response_metadata = AIModelCallResponseMetaData(streaming=True)

        # TODO: cleanup naming
        self.provider_metadata = {}
        self.message_metadata = {}  # Anthropic
        self.anthropic_tool_use_indices = set()
        self.persistant_choice_metadata_list = []  # OpenAI

        # Accumulated final response coming out of SDKs natively.
        # When set, we prefer this response over incremental reconstruction.
        self.native_final_response_dai: ChatResponse | None = None
        self.native_final_response_sdk: dict | None = None

        start_time = datetime_type.now()
        # TODO_FUTURE: Create progress per choices ?
        self.progress = INTStreamingProgress(
            start_time=start_time,
            last_token_time=start_time,
        )

    def update_usage(self, usage: ChatResponseUsage | None = None):
        """Update usgae"""
        if usage:
            self.usage = usage

    def complete(self) -> AIModelCallResponse:
        """Mark streaming as complete and set final stats"""
        self.progress.is_complete = True

        # Calculate duration
        duration = self.progress.last_token_time - self.progress.start_time
        duration_seconds = duration.total_seconds()

        self.response_metadata.duration_seconds = duration_seconds
        self.response_metadata.provider_metadata = self.provider_metadata

        return self.get_final_response()

    def get_final_response(self) -> AIModelCallResponse:
        """Convert streaiming progress to ChatResponse format"""
        # If a provider supplied a native final response, honor it to ensure
        # exact fidelity (including provider_response and structured content).
        if self.native_final_response_dai is not None:
            api_call_status = ExternalApiCallStatus(
                status=ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS,
                model=self.model_endpoint.ai_model.model_name,
                api_provider=self.model_endpoint.api.provider,
                message="Streaming Completed",
                code="success",
                http_status_code=200,
            )

            return AIModelCallResponse(
                status=api_call_status,
                chat_response=self.native_final_response_dai,
                image_response=None,
            )

        # Fall back to legacy reconstruction
        chat_response = None

        # If structured output was requested, derive it from accumulated text items
        try:
            if self.structured_output_config is not None:
                for choice in self.choices or []:
                    # Track indices of text items to remove after replacement
                    items_to_remove = []
                    items_to_add = []

                    for i, content in enumerate(choice.contents or []):
                        # Reasoning should not be parsed as structured output
                        if content.type == ChatResponseContentItemType.REASONING:
                            continue

                        # For Anthropic: structured output is delivered as tool_use blocks
                        # Convert ChatResponseToolCallContentItem to ChatResponseStructuredOutputContentItem
                        if content.type == ChatResponseContentItemType.TOOL_CALL:
                            if content.tool_call:
                                # CRITICAL: Build the complete raw_response structure for round-tripping
                                # This is especially important for Anthropic where structured output
                                # needs to be converted back to tool_use block format with id/name/type/input
                                # This ensure message conversion back to anthropic is unifrom across
                                # streaming and non-streaming cases
                                raw_resp_synth = {
                                    "type": "tool_use",
                                    "id": content.tool_call.call_id,
                                    "name": content.tool_call.name,
                                    "input": content.tool_call.arguments,
                                }

                                structured = ChatResponseStructuredOutput.from_tool_call(
                                    raw_response=raw_resp_synth,
                                    tool_call=content.tool_call,
                                    config=self.structured_output_config,
                                )
                                items_to_add.append(
                                    ChatResponseStructuredOutputContentItem(
                                        index=content.index,
                                        type=ChatResponseContentItemType.STRUCTURED_OUTPUT,
                                        role=getattr(content, "role", None),
                                        message_id=getattr(content, "message_id", None),
                                        message_contents=getattr(content, "message_contents", None),
                                        structured_output=structured,
                                    )
                                )
                                items_to_remove.append(i)
                            continue

                        # For OpenAI/Google: structured output is delivered as text
                        # Convert ChatResponseTextContentItem to ChatResponseStructuredOutputContentItem
                        if content.type == ChatResponseContentItemType.TEXT:
                            raw_text = content.get_text()
                            if raw_text is None:
                                continue

                            # CRITICAL FIX: Only attempt to parse text that looks like JSON.
                            # With reasoning models, we may get multiple text items:
                            # 1. Reasoning/thinking text (e.g., "**Crafting JSON...**")
                            # 2. The actual JSON output (e.g., '{"key": "value"}')
                            # We should only parse items that start with { or [ after stripping.
                            stripped = raw_text.strip()
                            if not (stripped.startswith("{") or stripped.startswith("[")):
                                # This is likely reasoning text or other non-JSON content
                                # Skip parsing and leave as-is
                                logger.debug(
                                    f"Skipping structured output parsing for text item {i} (does not look like JSON)"
                                )
                                continue

                            parsed_data, error, post_processed = ChatResponseStructuredOutput._parse_and_validate(
                                raw_data=raw_text,
                                config=self.structured_output_config,
                            )
                            # Always create a structured_output item to reflect parsing outcome
                            structured = ChatResponseStructuredOutput(
                                config=self.structured_output_config,
                                structured_data=parsed_data,
                                raw_data=(None if parsed_data is not None else raw_text),
                                parse_error=error,
                                post_processed=post_processed,
                            )
                            # Replace the text item with structured item, inheriting provider metadata.
                            # Rationale: structured output IS the text content in validated form.
                            # We keep message_id/message_contents from original text item for proper round-tripping.
                            # This matches non-streaming behavior where structured replaces text, not appends.
                            items_to_add.append(
                                ChatResponseStructuredOutputContentItem(
                                    index=content.index,
                                    type=ChatResponseContentItemType.STRUCTURED_OUTPUT,
                                    role=getattr(content, "role", None),
                                    message_id=getattr(content, "message_id", None),
                                    message_contents=getattr(content, "message_contents", None),
                                    structured_output=structured,
                                )
                            )
                            items_to_remove.append(i)

                    # Remove text items in reverse order to maintain indices
                    for i in reversed(items_to_remove):
                        choice.contents.pop(i)

                    # Add structured items
                    choice.contents.extend(items_to_add)
        except Exception as _e:
            logger.debug(f"Structured-output post-processing skipped due to error: {_e}")

        usage, usage_charge = self.get_streaming_usage_and_charge()

        if self.model_endpoint.ai_model.functional_type == AIModelFunctionalTypeEnum.TEXT_GENERATION:
            chat_response = ChatResponse(
                model=self.model_endpoint.ai_model.model_name,
                provider=self.model_endpoint.ai_model.provider,
                api_provider=self.model_endpoint.api.provider,
                usage=usage,
                usage_charge=usage_charge,
                choices=self.choices,
                metadata=self.response_metadata,
            )
        else:
            logger.fatal("Streaming is only supported for Chat generation models")
            return AIModelCallResponse(
                status=ExternalApiCallStatus(
                    status=ExternalApiCallStatusEnum.INTERNAL_PROCESSING_ERROR,
                    model=self.model_endpoint.ai_model.model_name,
                    api_provider=self.model_endpoint.api.provider,
                    message=(
                        f"Model {self.model_endpoint.ai_model.model_name} not supported for streaming. "
                        "Only Chat models are supported."
                    ),
                    code="error",
                    http_status_code=400,
                ),
            )

        api_call_status = ExternalApiCallStatus(
            status=ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS,
            model=self.model_endpoint.ai_model.model_name,
            api_provider=self.model_endpoint.api.provider,
            message="Streaming Completed",
            code="success",
            http_status_code=200,
        )

        return AIModelCallResponse(
            status=api_call_status,
            chat_response=chat_response,
            image_response=None,
        )

    def get_streaming_done_chunk(self):
        return StreamingChatResponse(
            id=None,
            data=ChatResponseChunk(
                model=self.model_endpoint.ai_model.model_name,
                provider=self.model_endpoint.ai_model.provider,
                api_provider=self.model_endpoint.api.provider,
                done=True,
            ),
        )

    def get_streaming_usage_and_charge(
        self,
    ) -> tuple[
        ChatResponseUsage | ImageResponseUsage | None,
        UsageCharge | None,
    ]:
        """Parse the OpenAI response into our standard format"""
        usage_charge = None

        if settings.ENABLE_USAGE_TRACKING or settings.ENABLE_COST_TRACKING:
            if not self.usage:
                logger.error(
                    "Usage not set before completing streaming. "
                    "Stream may have ended early due to SDK incompatibility or API error."
                )
                return (None, None)

            if settings.ENABLE_COST_TRACKING:
                usage_charge = self.model_endpoint.calculate_usage_charge(self.usage)

        return (self.usage, usage_charge)

    def update(
        self,
        choice_deltas: list[ChatResponseChoiceDelta],
        response_metadata: dict | None = None,
    ) -> ChatResponseChunk:
        """Update streaming progress with new chunk of deltas"""
        # Update metadata if provided
        if response_metadata:
            self.response_metadata.update(response_metadata)

        # Update last token time
        self.progress.last_token_time = datetime_type.now()

        if settings.ENABLE_STREAMING_CONSOLIDATION and choice_deltas:
            # Initialize choices list if empty
            if not self.choices:
                self.choices = [ChatResponseChoice(index=i, contents=[]) for i in range(len(choice_deltas))]

            # Process each choice delta
            for choice_delta in choice_deltas:
                choice_index = choice_delta.index

                # Ensure we have enough choices initialized
                while len(self.choices) <= choice_index:
                    self.choices.append(ChatResponseChoice(index=len(self.choices), contents=[]))

                choice = self.choices[choice_index]

                # Update choice metadata
                if choice_delta.finish_reason is not None:
                    choice.finish_reason = choice_delta.finish_reason

                if choice_delta.metadata:
                    choice.metadata.update(choice_delta.metadata)

                # Process content deltas if any
                if choice_delta.content_deltas:
                    # Initialize contents list if empty
                    if not choice.contents:
                        choice.contents = []

                    for content_delta in choice_delta.content_deltas:
                        # Find matching content by type and index, or create new
                        matching_content = None

                        # First try to find exact match by type and index
                        for content in choice.contents:
                            if content and content.type == content_delta.type and content.index == content_delta.index:
                                matching_content = content
                                break

                        # If no exact match, try to find by type only
                        if not matching_content:
                            reversed_contents = list(reversed(choice.contents))
                            for content in reversed_contents:
                                if content and content.type == content_delta.type:
                                    matching_content = content
                                    break

                        # If still no match, create new content item
                        if not matching_content:
                            # Create new content based on delta type
                            if content_delta.type == ChatResponseContentItemType.TEXT:
                                message_id = content_delta.message_id
                                message_contents = content_delta.message_contents
                                # Always rely on message_contents; initialize with a single output_text part
                                # if provider sends only text_delta increments.
                                if not message_contents:
                                    message_contents = [
                                        ChatMessageContentPart(type="output_text", text="", annotations=None)
                                    ]
                                matching_content = ChatResponseTextContentItem(
                                    index=content_delta.index,
                                    type=ChatResponseContentItemType.TEXT,
                                    role=content_delta.role,
                                    message_id=message_id,
                                    message_contents=message_contents,
                                    metadata=content_delta.metadata,
                                    storage_metadata=content_delta.storage_metadata,
                                    custom_metadata=content_delta.custom_metadata,
                                )
                            elif content_delta.type == ChatResponseContentItemType.REASONING:
                                # Initialize reasoning item; summary is list[ChatMessageContentPart] or None
                                thinking_id = content_delta.thinking_id
                                # Legacy hack removed: do NOT pull summary from metadata

                                message_contents = None
                                thinking_summary = None
                                if content_delta.text_delta is not None:
                                    message_contents = [
                                        ChatMessageContentPart(type="thinking", text="", annotations=None)
                                    ]
                                if content_delta.thinking_summary_delta is not None:
                                    thinking_summary = [
                                        ChatMessageContentPart(type="summary_text", text="", annotations=None)
                                    ]

                                matching_content = ChatResponseReasoningContentItem(
                                    index=content_delta.index,
                                    type=ChatResponseContentItemType.REASONING,
                                    role=content_delta.role,
                                    thinking_id=thinking_id,
                                    message_contents=message_contents,
                                    thinking_summary=thinking_summary,
                                    metadata=content_delta.metadata,
                                    storage_metadata=content_delta.storage_metadata,
                                    custom_metadata=content_delta.custom_metadata,
                                )
                            elif content_delta.type == ChatResponseContentItemType.TOOL_CALL:
                                # Create tool-call item with a placeholder to satisfy validation
                                # Prefer name from delta.tool_call if present; else from metadata hint; else 'unknown'
                                _name = None
                                if hasattr(content_delta, "tool_call") and content_delta.tool_call:
                                    _name = content_delta.tool_call.name
                                if not _name:
                                    _name = (
                                        (content_delta.metadata or {}).get("name")
                                        or (content_delta.metadata or {}).get("tool_name_delta")
                                        or "unknown"
                                    )

                                # Extract any known identifiers from metadata (e.g., OpenAI Responses)
                                _call_id = None
                                _item_id = None
                                if content_delta.metadata:
                                    _call_id = content_delta.metadata.get("call_id")
                                    _item_id = content_delta.metadata.get("item_id")

                                matching_content = ChatResponseToolCallContentItem(
                                    index=content_delta.index,
                                    type=ChatResponseContentItemType.TOOL_CALL,
                                    role=content_delta.role,
                                    tool_call=ChatResponseToolCall(
                                        call_id=(
                                            content_delta.tool_call.call_id
                                            if getattr(content_delta, "tool_call", None)
                                            else _call_id
                                        ),
                                        id=(
                                            content_delta.tool_call.id
                                            if getattr(content_delta, "tool_call", None)
                                            else _item_id
                                        ),
                                        name=_name,
                                        arguments={},
                                    ),
                                    metadata=content_delta.metadata,
                                    storage_metadata=content_delta.storage_metadata,
                                    custom_metadata=content_delta.custom_metadata,
                                )
                            elif content_delta.type == ChatResponseContentItemType.GENERIC:
                                matching_content = ChatResponseGenericContentItem(
                                    index=content_delta.index,
                                    type=ChatResponseContentItemType.GENERIC,
                                    role=content_delta.role,
                                    metadata=content_delta.metadata,
                                    storage_metadata=content_delta.storage_metadata,
                                    custom_metadata=content_delta.custom_metadata,
                                )
                            else:
                                logger.error(f"stream_manager: Unknown content_delta type {content_delta.type}")
                                continue

                            choice.contents.append(matching_content)

                        # Verify type matches
                        if matching_content.type != content_delta.type or matching_content.index != content_delta.index:
                            logger.error(f"stream_manager: Content type mismatch at index {content_delta.index}")
                            continue

                        # Update content based on delta type
                        if content_delta.type == ChatResponseContentItemType.TEXT:
                            # Otherwise, append raw text_delta into the first output_text part
                            if content_delta.text_delta:
                                if not matching_content.message_contents:
                                    matching_content.message_contents = [
                                        ChatMessageContentPart(type="output_text", text="", annotations=None)
                                    ]
                                # Ensure we have at least one writable text part
                                if not matching_content.message_contents:
                                    matching_content.message_contents.append(
                                        ChatMessageContentPart(type="output_text", text="", annotations=None)
                                    )
                                # Append to the first text-bearing part
                                matching_content.message_contents[0].text = (
                                    matching_content.message_contents[0].text or ""
                                ) + content_delta.text_delta

                        elif content_delta.type == ChatResponseContentItemType.REASONING:
                            # Append reasoning deltas into message_contents or thinking_summary list parts
                            thinking_id = content_delta.thinking_id
                            thinking_signature = content_delta.thinking_signature

                            # Reasoning text goes into message_contents parts
                            if content_delta.text_delta:
                                if not matching_content.message_contents:
                                    matching_content.message_contents = [
                                        ChatMessageContentPart(type="thinking", text="", annotations=None)
                                    ]
                                matching_content.message_contents[0].text = (
                                    matching_content.message_contents[0].text or ""
                                ) + content_delta.text_delta

                            # Streamed thinking summary accumulation
                            elif content_delta.thinking_summary_delta:
                                if not matching_content.thinking_summary:
                                    matching_content.thinking_summary = [
                                        ChatMessageContentPart(type="summary_text", text="", annotations=None)
                                    ]

                                # Append to first thinking part
                                matching_content.thinking_summary[0].text = (
                                    matching_content.thinking_summary[0].text or ""
                                ) + content_delta.thinking_summary_delta

                            else:
                                logger.warning(
                                    "stream_manager: reasoning delta without text or summary; ignoring incremental"
                                )

                            if thinking_id:
                                matching_content.thinking_id = thinking_id
                            if thinking_signature:
                                matching_content.thinking_signature = thinking_signature

                        elif content_delta.type in (
                            ChatResponseContentItemType.TOOL_CALL,
                            ChatResponseContentItemType.GENERIC,
                        ):
                            # Update metadata for tool calls and generic content
                            matching_content.metadata.update(content_delta.metadata)

                            # If it's a tool call, update the incremental arguments, name, or set full tool_call
                            if content_delta.type == ChatResponseContentItemType.TOOL_CALL:
                                if hasattr(content_delta, "tool_call") and content_delta.tool_call:
                                    # If we have a complete tool_call object, set/replace it
                                    if hasattr(matching_content, "tool_call"):
                                        matching_content.tool_call = content_delta.tool_call
                                # Update name from metadata deltas if present
                                _md = content_delta.metadata or {}
                                if _md.get("name") and getattr(matching_content, "tool_call", None):
                                    matching_content.tool_call.name = _md.get("name")
                                if _md.get("tool_name_delta") and getattr(matching_content, "tool_call", None):
                                    # Accumulate piecewise name in a buffer
                                    name_buf = matching_content.metadata.get("name_buffer", "") + _md.get(
                                        "tool_name_delta"
                                    )
                                    matching_content.metadata["name_buffer"] = name_buf
                                    try:
                                        matching_content.tool_call.name = name_buf
                                    except Exception:
                                        pass
                                # Handle incremental arguments appends into metadata buffer
                                if hasattr(content_delta, "arguments_delta") and content_delta.arguments_delta:
                                    # Maintain a buffer for args in metadata
                                    buf_key = "arguments_buffer"
                                    prev = matching_content.metadata.get(buf_key) or ""
                                    matching_content.metadata[buf_key] = prev + content_delta.arguments_delta

                                # Finalize tool call arguments when signaled
                                if hasattr(content_delta, "metadata") and content_delta.metadata.get(
                                    "finalize_tool_call"
                                ):
                                    buf_key = "arguments_buffer"
                                    raw_buf = matching_content.metadata.get(buf_key)
                                    if raw_buf is not None:
                                        try:
                                            import json as _json

                                            parsed = _json.loads(raw_buf)
                                            parse_error = None
                                        except Exception as e:
                                            parsed = {}
                                            parse_error = str(e)

                                        # Ensure tool_call exists
                                        if not getattr(matching_content, "tool_call", None):
                                            # Create a placeholder tool_call
                                            matching_content.tool_call = ChatResponseToolCall(
                                                call_id=None,
                                                id=None,
                                                name=matching_content.metadata.get("name") or "unknown",
                                                arguments={},
                                            )

                                        # Assign parsed args and parse error
                                        matching_content.tool_call.arguments = (
                                            parsed if isinstance(parsed, dict) else {"raw": raw_buf}
                                        )
                                        if parse_error:
                                            matching_content.tool_call.parse_error = parse_error

                                        # Clear buffer
                                        try:
                                            del matching_content.metadata[buf_key]
                                        except Exception:
                                            matching_content.metadata[buf_key] = ""

        # Update token count
        self.progress.updates_count += 1

        # Create and return stream chunk
        return ChatResponseChunk(
            model=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            usage=self.usage,
            usage_charge=self.usage_charge,
            choice_deltas=choice_deltas,
            metadata=self.response_metadata,
            done=False,
        )
