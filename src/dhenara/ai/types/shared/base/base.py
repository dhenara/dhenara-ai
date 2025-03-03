from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, alias_generators

T = TypeVar("T", bound="BaseModel")
# logger.debug(f"Pydantic version: {pydantic.__version__}")


class BaseEnum(str, Enum):
    """Base Enumeration class."""

    def __str__(self):
        return self.value

    @classmethod
    def values(cls) -> set[str]:
        """Get all values.

        Returns:
            set[str]: Set of all values
        """
        return {member.value for member in cls}


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=set(),
        # Enable detailed validation errors:
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        # schema etra
        json_schema_extra={"examples": []},
        str_strip_whitespace=False,  # Don't set: Streaming responses will be terrible
        use_enum_values=True,
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        dump_kwargs = {
            "exclude_unset": False,
            "by_alias": False,
            "exclude_none": True,
            "round_trip": False,
        }
        dump_kwargs.update(kwargs)

        return super().model_dump(**dump_kwargs)

    # def copy_with_changes(self: T, **changes) -> T:
    #    """Create a copy with specified changes."""
    #    data = self.model_dump()
    #    data.update(changes)
    #    return self.__class__.model_validate(data)

    # @classmethod
    # def safe_parse(cls: Type[T], data: dict[str, Any]) -> tuple[Optional[T], Optional[ValidationError]]:
    #    """Safely parse data without raising exceptions."""
    #    try:
    #        return cls.model_validate(data), None
    #    except ValidationError as e:
    #        return None, e
