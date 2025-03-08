from typing import Generic, TypeVar

# -----------------------------------------------------------------------------
from pydantic import Field

from dhenara.ai.types.shared.base import BaseEnum, BaseModel


# -----------------------------------------------------------------------------
class ApiRequestActionTypeEnum(BaseEnum):
    """Common action types for endpoint operations."""

    # GET
    list = "list"
    get = "get"
    # POST
    create = "create"
    update = "update"
    delete = "delete"
    activate = "activate"
    deactivate = "deactivate"
    # Run
    run = "run"


# -----------------------------------------------------------------------------
T = TypeVar("T", bound=BaseModel)


class ApiRequest(BaseModel, Generic[T]):
    """
    Base arequest model for generic operations.

    Args:
        data: The actual data for the action
        action: The type of action to perform
    """

    data: T
    action: ApiRequestActionTypeEnum | None = Field(
        ...,
        description="Action to perform on the endpoint",
    )
    custom_action: str | None = Field(
        default=None,
        description="Custom Action to perform on the endpoint",
    )
