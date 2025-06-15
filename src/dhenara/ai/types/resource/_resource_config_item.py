from pydantic import Field, model_validator

from dhenara.ai.types.shared import BaseEnum, BaseModel


class ResourceConfigItemTypeEnum(BaseEnum):
    """Enumeration of available resource model types."""

    ai_model_endpoint = "ai_model_endpoint"
    rag_endpoint = "rag_endpoint"
    search_endpoint = "search_endpoint"


class ResourceQueryFieldsEnum(BaseEnum):
    """Enum defining all possible query fields for resources."""

    model_name = "model_name"
    model_display_name = "model_display_name"
    api_provider = "api_provider"
    reference_number = "reference_number"


class ResourceQueryMapping:
    """Static Class defining the mapping between resource types and their allowed query fields."""

    MAPPINGS: dict[ResourceConfigItemTypeEnum, list[ResourceQueryFieldsEnum]] = {
        ResourceConfigItemTypeEnum.ai_model_endpoint: [
            ResourceQueryFieldsEnum.model_name,
            ResourceQueryFieldsEnum.model_display_name,
            ResourceQueryFieldsEnum.api_provider,
        ],
    }

    @classmethod
    def get_allowed_fields(cls, resource_type: ResourceConfigItemTypeEnum) -> list[str]:
        """
        Get allowed query fields for a resource type.

        Args:
            resource_type: The type of resource

        Returns:
            List of allowed field names
        """
        return [field.value for field in cls.MAPPINGS.get(resource_type, [])]


class ResourceConfigItem(BaseModel):
    """
    ResourceConfigItem configuration model with mutually exclusive fields for object parameters
    or fetch query.

    Attributes:
        item_type: Type of the resource model
        query: Optional query string for fetching resource details
        is_default: Flag to mark default resource
    """

    item_type: ResourceConfigItemTypeEnum = Field(
        ...,
        description="Type of the resource item",
    )
    query: dict | None = Field(
        default=None,
        description="Query dict or list of query dicts for fetching resource details",
        examples=[
            {"model_name": "claude-sonet-3.5-v2"},
            {
                "model_name": "claude-sonet-3.5-v2",
                "api_provider": "anthropic",
            },
        ],
    )
    is_default: bool = Field(
        default=False,
        description="Is default resource or not. Only one default is allowed in a list of resources",
    )

    @model_validator(mode="after")
    def validate_exclusive_fields(self) -> "ResourceConfigItem":
        """Validates mutually exclusive fields and query structure."""
        # Validate query keys based on model type
        for key in self.query.keys():
            allowed_fields = ResourceQueryMapping.get_allowed_fields(self.item_type)
            if key not in allowed_fields:
                raise ValueError(f"Unsupported query key `{key}`")

        return self

    def is_same_as(self, other: "ResourceConfigItem") -> bool:
        """
        Compares two ResourceConfigItem objects for equality based on their type and identifiers.

        Args:
            other: Another ResourceConfigItem object to compare with

        Returns:
            bool: True if the resources represent the same entity, False otherwise
        """
        # First check if object types match
        if self.item_type != other.item_type:
            return False

        # Case 2: Both have query
        if self.query and other.query:
            return self.query == other.query

        # Case 3: Mixed cases are considered different
        return False

    # Factory methods
    @classmethod
    def with_model(cls, model_name: str) -> list["ResourceConfigItem"]:
        return [
            cls(
                item_type=ResourceConfigItemTypeEnum.ai_model_endpoint,
                query={ResourceQueryFieldsEnum.model_name: model_name},
                is_default=True,
            )
        ]

    @classmethod
    def with_models(cls, model_names: list[str]) -> list["ResourceConfigItem"]:
        return [
            cls(
                item_type=ResourceConfigItemTypeEnum.ai_model_endpoint,
                query={ResourceQueryFieldsEnum.model_name: model_name},
                is_default=True if index == 0 else False,
            )
            for index, model_name in enumerate(model_names)
        ]

    @classmethod
    def get_model_names(cls, resources: list["ResourceConfigItem"]) -> list[str]:
        """
        Extracts all model names from a list of ResourceConfigItem objects.
        This is the reverse operation of `with_models`.

        Args:
            resources: List of ResourceConfigItem objects

        Returns:
            list[str]: List of model names
        """
        return [
            resource.query[ResourceQueryFieldsEnum.model_name]
            for resource in resources
            if (
                resource.item_type == ResourceConfigItemTypeEnum.ai_model_endpoint
                and resource.query
                and ResourceQueryFieldsEnum.model_name in resource.query
            )
        ]

    @classmethod
    def has_model(cls, resources: list["ResourceConfigItem"], model_name: str) -> bool:
        """
        Check whether a given model name is present in the supplied
        list of ResourceConfigItem objects.

        Args:
            resources: List of ResourceConfigItem objects
            model_name: The model name to look for

        Returns:
            bool: True if the model name exists, False otherwise
        """
        return model_name in cls.get_model_names(resources)
