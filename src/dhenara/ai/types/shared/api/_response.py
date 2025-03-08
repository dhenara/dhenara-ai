from typing import Any, Generic, TypeVar

from pydantic import Field

from dhenara.ai.types.shared.base import BaseEnum, BaseModel
from dhenara.ai.types.shared.platform import DhenaraAPIError


# -----------------------------------------------------------------------------
class ApiResponseStatus(BaseEnum):
    SUCCESS = "success"
    FAIL = "fail"
    ERROR = "error"
    PENDING = "pending"


# -----------------------------------------------------------------------------
class ApiResponseMessageStatusCode(BaseEnum):
    SUCCESSFUL = "successful"
    PENDING = "pending"
    INVALID_INPUTS = "invalid_inputs"
    FAIL_ENDPOINT_ERROR = "failed_with_endpoint_error"
    FAIL_SERVER_ERROR = "failed_with_server_error"
    NOT_AUTHENTICATED = "not_authenticated"
    NOT_AUTHORIZED = "not_authorized"
    PERMISSION_DENIED_GENERAL = "permission_denied"
    PERMISSION_DENIED_BY_WORKSPACE = "permission_denied_by_workspace"
    USAGE_EXCEEDED_LIMIT = "usage_exceeded_limit"


# -----------------------------------------------------------------------------
class ApiResponseMessageType(BaseEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


# -----------------------------------------------------------------------------
class ApiResponseMessage(BaseModel):
    type: ApiResponseMessageType
    status_code: ApiResponseMessageStatusCode
    message: str = ""
    message_data: dict | None = None
    show: bool = False  # Whether to show this message to user in case of FE

    def is_successful(self) -> bool:
        return self.status_code in [
            ApiResponseMessageStatusCode.SUCCESSFUL,
            ApiResponseMessageStatusCode.PENDING,
        ]


# -----------------------------------------------------------------------------
T = TypeVar("T", bound=BaseModel)


# -----------------------------------------------------------------------------
class ApiResponse(BaseModel, Generic[T]):
    status: ApiResponseStatus
    messages: list[ApiResponseMessage] = Field(default_factory=list)
    data: T | dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        return self.status == ApiResponseStatus.SUCCESS

    @property
    def is_pending(self) -> bool:
        return self.status == ApiResponseStatus.PENDING

    @property
    def first_message(self) -> ApiResponseMessage | None:
        return self.messages[0] if self.messages else None

    def check_for_status_errors(self) -> None:
        """Raises an exception if the response indicates an error"""
        if not self.is_success:
            if self.first_message:
                return self.first_message
            return "Unknown error occurred"
        return None

    def raise_for_status(self) -> None:
        """Raises an exception if the response indicates an error"""
        error_msg = self.check_for_status_errors()
        if error_msg:
            raise DhenaraAPIError(
                message="Unknown error occurred",
                status_code=self.first_message.status_code or ApiResponseMessageStatusCode.FAIL_SERVER_ERROR,
                response={},
            )
