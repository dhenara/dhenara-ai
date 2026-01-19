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
# Add the missing status codes to ApiResponseMessageStatusCode
class ApiResponseMessageStatusCode(BaseEnum):
    SUCCESSFUL = "successful"
    PENDING = "pending"
    INVALID_INPUTS = "invalid_inputs"
    FAIL_ENDPOINT_ERROR = "failed_with_endpoint_error"
    FAIL_SERVER_ERROR = "failed_with_server_error"
    FAIL_FORBIDDEN = "failed_forbidden"
    FAIL_BAD_REQUEST = "failed_with_bad_request"
    NOT_AUTHENTICATED = "not_authenticated"
    NOT_AUTHORIZED = "not_authorized"
    PERMISSION_DENIED_GENERAL = "permission_denied"
    PERMISSION_DENIED_BY_WORKSPACE = "permission_denied_by_workspace"
    USAGE_EXCEEDED_LIMIT = "usage_exceeded_limit"
    # Auth-related status codes
    AUTH_MISSING_CREDENTIALS = "auth_missing_credentials"
    AUTH_INVALID_CREDENTIALS = "auth_invalid_credentials"
    AUTH_ACCOUNT_INACTIVE = "auth_account_inactive"
    AUTH_LOGIN_SUCCESS = "auth_login_success"
    AUTH_INVALID_REQUEST = "auth_invalid_request"
    AUTH_SERVER_ERROR = "auth_server_error"
    AUTH_MISSING_REFRESH_TOKEN = "auth_missing_refresh_token"
    AUTH_INVALID_REFRESH_TOKEN = "auth_invalid_refresh_token"
    AUTH_TOKEN_REFRESHED = "auth_token_refreshed"
    AUTH_LOGOUT_SUCCESS = "auth_logout_success"

    # Opps
    FAIL_NOT_FOUND = "fail_not_found"


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
class ApiResponse(BaseModel, Generic[T]):  # noqa: UP046
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

    def check_for_status_errors(self) -> ApiResponseMessage | str | None:
        """Return an error object/message when status indicates failure."""
        if self.is_success:
            return None
        if self.first_message:
            return self.first_message
        return "Unknown error occurred"

    def raise_for_status(self) -> None:
        """Raises an exception if the response indicates an error"""
        error_msg = self.check_for_status_errors()
        if error_msg:
            message = error_msg.message if isinstance(error_msg, ApiResponseMessage) else str(error_msg)
            fm = self.first_message

            code = (
                fm.status_code
                if (fm is not None and fm.status_code is not None)
                else ApiResponseMessageStatusCode.FAIL_SERVER_ERROR
            )
            http_status: int | None = 500
            if code in {ApiResponseMessageStatusCode.INVALID_INPUTS, ApiResponseMessageStatusCode.FAIL_BAD_REQUEST}:
                http_status = 400
            elif code in {
                ApiResponseMessageStatusCode.NOT_AUTHENTICATED,
                ApiResponseMessageStatusCode.AUTH_MISSING_CREDENTIALS,
                ApiResponseMessageStatusCode.AUTH_INVALID_CREDENTIALS,
                ApiResponseMessageStatusCode.AUTH_INVALID_REQUEST,
            }:
                http_status = 401
            elif code in {
                ApiResponseMessageStatusCode.NOT_AUTHORIZED,
                ApiResponseMessageStatusCode.FAIL_FORBIDDEN,
                ApiResponseMessageStatusCode.PERMISSION_DENIED_GENERAL,
                ApiResponseMessageStatusCode.PERMISSION_DENIED_BY_WORKSPACE,
            }:
                http_status = 403
            raise DhenaraAPIError(
                message=message,
                status_code=http_status,
                response={},
            )
