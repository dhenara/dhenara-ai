import json
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import Field

from dhenara.ai.types.shared.base import BaseEnum, BaseModel

# Type variable for generic data types
T = TypeVar("T", bound=BaseModel)


class SSEEventType(BaseEnum):
    """Types of Server-Sent Events"""

    TOKEN_STREAM = "token_stream"  # Streaming content chunks
    PUSH = "push"  # Push notifications
    ERROR = "error"  # Error events


class SSEErrorCode(BaseEnum):
    server_error = "server_error"
    external_api_error = "external_api_error"
    client_decode_error = "client_decode_error"


class SSEErrorData(BaseModel):
    """Model for error response data"""

    error_code: SSEErrorCode = Field(
        ...,
        description="Error code identifier",
    )
    message: str = Field(
        ...,
        description="Error message",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )


class SSEResponse(BaseModel, Generic[T]):
    """Generic SSE response model supporting different data types"""

    event: SSEEventType = Field(
        ...,
        description="SSE Event type",
    )
    data: T = Field(
        ...,
        description="Event data payload",
    )
    id: str | None = Field(
        default=lambda: None,
        # default_factory=lambda: str(uuid4()),
        description="Unique event identifier",
    )
    retry: int | None = Field(
        default=None,
        description="Retry timeout in milliseconds",
    )

    class Config:
        arbitrary_types_allowed = True

    def set_random_id(self) -> str:
        self.id = str(uuid4())

    def is_error(self) -> bool:
        return self.event == SSEEventType.ERROR

    def to_sse_format(self) -> str:
        """Convert to SSE format string"""
        lines = []

        # Add event type
        lines.append(f"event: {self.event}")

        # Add ID
        lines.append(f"id: {self.id}")

        # Add retry if present
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")

        # Add data
        if isinstance(self.data, BaseModel):
            data_str = self.data.model_dump_json()
        elif isinstance(self.data, (dict, list)):
            data_str = json.dumps(self.data)
        else:
            data_str = str(self.data)

        # Handle multi-line data
        for line in data_str.splitlines():
            lines.append(f"data: {line}")  # noqa: PERF401

        return "\n".join(lines) + "\n\n"

    @classmethod
    def parse_sse(cls, sse_str: str, data_type: T | None = None) -> "SSEResponse[Any]":
        """Parse SSE format string into response object"""
        lines = sse_str.strip().split("\n")
        event_data = {
            "event": None,
            "id": None,
            "retry": None,
            "data": None,
        }

        data_lines = []

        for line in lines:
            if not line.strip():
                continue

            if ":" not in line:
                continue

            field, value = line.split(":", 1)
            field = field.strip()
            value = value.lstrip()

            if field == "data":
                data_lines.append(value)
            elif field in event_data:
                event_data[field] = value

        # Parse event type first
        try:
            event_type = SSEEventType(event_data["event"]) if event_data["event"] else SSEEventType.ERROR
        except ValueError:
            return SSEErrorResponse(
                data=SSEErrorData(
                    error_code=SSEErrorCode.client_decode_error,
                    message=f"Invalid event type: {event_data['event']}",
                ),
            )

        # Join and parse data
        if data_lines:
            try:
                raw_data = json.loads("".join(data_lines))

                # Handle different event types
                if event_type == SSEEventType.ERROR:
                    data = SSEErrorData(
                        error_code=SSEErrorCode(raw_data.get("error_code", SSEErrorCode.server_error)),
                        message=raw_data.get("message", "Unknown error"),
                        details=raw_data.get("details"),
                    )
                else:
                    # For normal events, create T object
                    if data_type is not None:
                        data = data_type(**raw_data)
                    else:
                        data = raw_data

            except json.JSONDecodeError:
                return SSEErrorResponse(
                    data=SSEErrorData(
                        error_code=SSEErrorCode.client_decode_error,
                        message="Invalid JSON in stream data",
                    ),
                )
            except Exception as e:
                return SSEErrorResponse(
                    data=SSEErrorData(
                        error_code=SSEErrorCode.client_decode_error,
                        message=f"Failed to parse stream data: {e!s}",
                    ),
                )
        else:
            return SSEErrorResponse(
                data=SSEErrorData(
                    error_code=SSEErrorCode.client_decode_error,
                    message="No data found in SSE event",
                ),
            )

        # Create response with proper typing
        try:
            if event_type == SSEEventType.ERROR:
                return SSEErrorResponse(
                    data=data,
                    id=event_data["id"],
                    retry=int(event_data["retry"]) if event_data["retry"] else None,
                )
            else:
                return cls(
                    event=event_type,
                    data=data,
                    id=event_data["id"],
                    retry=int(event_data["retry"]) if event_data["retry"] else None,
                )
        except Exception as e:
            return SSEErrorResponse(
                data=SSEErrorData(
                    error_code=SSEErrorCode.client_decode_error,
                    message=f"Failed to create SSE response: {e!s}",
                ),
            )


class SSEErrorResponse(SSEResponse[SSEErrorData]):
    """Specialized SSE response for chat streaming"""

    event: SSEEventType = SSEEventType.ERROR
    data: SSEErrorData
