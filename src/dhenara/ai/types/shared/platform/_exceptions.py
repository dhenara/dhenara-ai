class DhenaraError(Exception):
    """Base exception for Dhenara API errors"""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class DhenaraAPIError(DhenaraError):
    """Raised when the API returns an error response"""

    pass


class DhenaraConnectionError(DhenaraError):
    """Raised when there's a network connection error"""

    pass


# class ValidationError(APIError):
#    """Raised when the request data is invalid."""
#    pass
#
#
# class AuthenticationError(APIError):
#    """Raised when authentication fails."""
#    pass
#
#
# class PermissionError(APIError):
#    """Raised when the user doesn't have permission."""
#    pass
#
#
# class NotFoundError(APIError):
#    """Raised when a resource is not found."""
#    pass
