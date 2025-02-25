from enum import Enum
from typing import Any, Literal, TypeVar

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
    """Base model class with enhanced configuration and serialization capabilities.

    This base class provides:
    - Camel case conversion for API interactions
    - Flexible serialization options
    - Support for arbitrary types
    - Strict validation by default
    """

    def model_dump(self, exclude: list | None = None) -> dict:
        result = super().model_dump(
            exclude=exclude,
            exclude_unset=True,
            # mode="json",
            by_alias=False,  # Not to use camel case in dict keys
            exclude_none=True,
            round_trip=False,
        )

        return result

    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=set(),
        extra="forbid",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": []  # can be overridden by child classes
        },
    )

    def to_dict(
        self: T,
        *,
        mode: Literal["json", "python"] = "python",
        use_api_names: bool = True,
        exclude_unset: bool = True,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        warnings: bool = True,
    ) -> dict[str, Any]:
        """Convert the model to a dictionary representation.

        Args:
            mode: Serialization mode ('json' for JSON-compatible types, 'python' for Python objects)
            use_api_names: Use API response keys instead of property names
            exclude_unset: Exclude unset fields
            exclude_defaults: Exclude fields with default values
            exclude_none: Exclude None values
            warnings: Enable serialization warnings (Pydantic v2 feature)

        Returns:
            dict: Dictionary representation of the model

        Examples:
            ```python
            model_dict = model.to_dict(mode="python", exclude_none=True)
            ```
        """
        return self.model_dump(
            mode=mode,
            by_alias=use_api_names,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude=[None] if exclude_none else [],
            warnings=warnings,
        )

    def to_json(
        self: T,
        *,
        indent: int | None = 2,
        use_api_names: bool = True,
        exclude_unset: bool = True,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        warnings: bool = True,
    ) -> str:
        """Convert the model to a JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact output)
            use_api_names: Use API response keys instead of property names
            exclude_unset: Exclude unset fields
            exclude_defaults: Exclude fields with default values
            exclude_none: Exclude None values
            warnings: Enable serialization warnings (Pydantic v2 feature)

        Returns:
            str: JSON string representation of the model

        Examples:
            ```python
            json_str = model.to_json(indent=4, exclude_none=True)
            ```
        """
        return self.model_dump_json(
            indent=indent,
            by_alias=use_api_names,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            warnings=warnings,
        )

    @classmethod
    def schema_json(cls: type[T], by_alias: bool = True) -> str:
        """Get JSON schema for the model.

        Args:
            by_alias: Use API field names instead of Python field names

        Returns:
            str: JSON schema representation
        """
        return cls.model_json_schema(by_alias=by_alias)
