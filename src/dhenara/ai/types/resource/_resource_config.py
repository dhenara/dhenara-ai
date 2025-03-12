from pydantic import Field

from dhenara.ai.types.genai.ai_model import AIModelEndpoint
from dhenara.ai.types.resource._resource_config_item import ResourceConfigItem, ResourceConfigItemTypeEnum
from dhenara.ai.types.shared.base import BaseModel


class ResourceConfig(BaseModel):
    """Resources"""

    # TODO_FUTURE: Add apis
    # Also add a function to create new model endpoint after validating provider

    ai_model_endpoints: list[AIModelEndpoint] = Field(
        default_factory=list,
        description="AIModel Endpoins",
    )

    def get_resource(self, resource_item: ResourceConfigItem) -> AIModelEndpoint:  # |RagEndpoint
        """
        Retrieves a resource based on the resource specification.

        Args:
            resource: ResourceConfigItem model instance

        Returns:
            AIModelEndpoint instance

        Raises:
            ValueError: If object not found or query invalid
        """
        if not resource_item.query:
            raise ValueError("Query must be provided")

        try:
            if resource_item.item_type == ResourceConfigItemTypeEnum.ai_model_endpoint:
                # Filter based on query parameters
                for endpoint in self.ai_model_endpoints:
                    matches = True

                    for key, value in resource_item.query.items():
                        if key == "reference_number" and endpoint.reference_number != value:
                            matches = False
                            break
                        elif key == "model_name" and endpoint.ai_model.model_name != value:
                            matches = False
                            break
                        elif key == "model_display_name" and endpoint.ai_model.display_name != value:
                            matches = False
                            break
                        elif key == "api_provider" and endpoint.api.provider != value:
                            matches = False
                            break

                    if matches:
                        return endpoint

                # Create query description for error message
                query_desc = ", ".join(f"{k}={v}" for k, v in resource_item.query.items())
                raise ValueError(f"No endpoint found matching query: {query_desc}")

            else:
                raise ValueError(f"Unsupported resource type: {resource_item.item_type}")

        except Exception as e:
            raise ValueError(f"Error fetching resource: {e}")
