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

    # TODO_FUTURE: Comeup with better soution to mask selective fields
    def model_dump_safe(self, **kwargs):
        """Get a string representation with masked sensitive data"""
        # Get the regular model dump
        data = self.model_dump()
        # Create a copy to avoid modifying the original data
        return self._mask_sensitive_data(data.copy())

    def _mask_sensitive_data(self, data: dict) -> dict:
        """Recursively mask sensitive fields in a dictionary"""
        if not isinstance(data, dict):
            return data

        for key, value in data.items():
            if key == "api_key" and value:
                data[key] = f"{str(value)[:4]}...{str(value)[-4:]}" if len(str(value)) > 8 else value
            elif key == "credentials" and value and isinstance(value, dict):
                data[key] = {
                    k: f"{str(v)[:4]}...{str(v)[-4:]}" if v and len(str(v)) > 8 else v for k, v in value.items()
                }
            elif isinstance(value, dict):
                data[key] = self._mask_sensitive_data(value)
            elif isinstance(value, list):
                data[key] = [self._mask_sensitive_data(i) if isinstance(i, dict) else i for i in value]
        return data

    def _get_masked_representation(self) -> str:
        masked_data = self.model_dump_safe()

        # import json
        # return json.dumps(self.model_dump_safe(), indent=2)

        # Format it like the default __repr__ but with masked data
        fields = ", ".join(f"{k}={repr(v)}" for k, v in masked_data.items())  # noqa: RUF010
        return f"{self.__class__.__name__}({fields})"
