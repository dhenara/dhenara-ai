import json
import logging
import os
from pathlib import Path

import yaml
from pydantic import Field

from dhenara.ai.types.genai import MODEL_TO_API_MAPPING, PROVIDER_CONFIGS
from dhenara.ai.types.genai.ai_model import AIModel, AIModelAPI, AIModelEndpoint, FoundationModel
from dhenara.ai.types.genai.foundation_models import ALL_FOUNDATION_MODELS
from dhenara.ai.types.resource._resource_config_item import ResourceConfigItem, ResourceConfigItemTypeEnum
from dhenara.ai.types.shared.base import BaseModel

logger = logging.getLogger(__name__)


class ResourceConfig(BaseModel):
    """
    Configuration for AI resources including models, APIs, and endpoints.
    Manages loading credentials, initializing APIs, and creating model endpoints.
    """

    models: list[AIModel | FoundationModel] = Field(
        default_factory=list,
        description="AIModels",
    )
    model_apis: list[AIModelAPI] = Field(
        default_factory=list,
        description="AIModel APIs",
    )
    model_endpoints: list[AIModelEndpoint] = Field(
        default_factory=list,
        description="AIModel Endpoints",
    )

    def get_api(self, api_provider: str | None = None) -> AIModelEndpoint:
        return next((api for api in self.model_apis if api.provider == api_provider), None)

    def get_model_endpoint(self, model_name: str, api_provider: str | None = None) -> AIModelEndpoint:
        """
        Retrieves an endpoint by model name.

        Args:
            model_name: Model name

        Returns:
            AIModelEndpoint instance
        """

        query = {"model_name": model_name}
        if api_provider:
            query["api_provider"] = api_provider

        return self.get_resource(
            ResourceConfigItem(
                item_type=ResourceConfigItemTypeEnum.ai_model_endpoint,
                query=query,
            )
        )

    def get_resource(self, resource_item: ResourceConfigItem) -> AIModelEndpoint:  # |RagEndpoint
        """
        Retrieves a resource based on the resource specification.

        Args:
            resource_item: ResourceConfigItem model instance

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
                for endpoint in self.model_endpoints:
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
                logger.error(f"No endpoint found matching query: {query_desc}")
                return None

            else:
                logger.error(f"Unsupported resource type: {resource_item.item_type}")
                return None

        except Exception as e:
            logger.error(f"Error fetching resource: {e}")
            return None

    def load_from_file(
        self,
        credentials_file: str,
        models: list[AIModel] | None = None,
        init_endpoints: bool = False,
    ):
        """
        Initialize the ResourceConfig with credentials from a file and optional overrides.

        Args:
            credentials_file: Path to the credentials file (JSON or YAML)
            models: Optional list of models to override default ALL_FOUNDATION_MODELS
            mapping_override: Optional dictionary to override default model-to-API mappings
        """

        # Load credentials
        credentials = self._load_credentials_from_file(credentials_file)

        # Initialize APIs with loaded credentials
        self._initialize_apis(credentials)

        if models:
            self.models = models

        if init_endpoints:
            if not self.models:
                self.models = [*ALL_FOUNDATION_MODELS]

            # Create endpoints from models and APIs
            self._initialize_endpoints()

    def _load_credentials_from_file(self, credentials_file: str) -> dict:
        """
        Load credentials from a file (JSON or YAML format).

        Args:
            credentials_file: Path to the credentials file

        Returns:
            Dictionary containing credentials for each provider

        Raises:
            FileNotFoundError: If the credentials file doesn't exist
            ValueError: If the file format is unsupported
        """
        # Expand the tilde to home directory if present
        expanded_path = os.path.expanduser(credentials_file)
        path = Path(expanded_path)

        if not path.exists():
            raise FileNotFoundError(f"Credentials file not found: {credentials_file}")

        ext = path.suffix.lower()
        with open(path) as f:
            if ext == ".json":
                credentials = json.load(f)
            elif ext in [".yaml", ".yml"]:
                credentials = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}. Use .json, .yaml, or .yml")

        # Validate loaded credentials
        return credentials

    def _initialize_apis(self, credentials: dict) -> None:
        """
        Initialize API clients based on loaded credentials.

        Args:
            credentials: Dictionary of credentials by provider
        """
        for provider_str, provider_creds_data in credentials.items():
            try:
                new_api = AIModelAPI(
                    provider=provider_str,
                    api_key=provider_creds_data.get("api_key", None),
                    credentials=provider_creds_data.get("credentials", {}),
                    config=provider_creds_data.get("config", {}),
                )

                # If validation passes, add the endpint
                self.model_apis.append(new_api)

            except Exception as e:  # noqa: PERF203
                logger.exception(f"Error initializing API for provider '{provider_str}': {e}")

    def _initialize_endpoints(self) -> None:
        """
        Create model endpoints by matching models with compatible APIs.
        For each model, finds a compatible API and creates an endpoint.
        """
        # Track created endpoints to avoid duplicates
        created_endpoints = set()

        for model in self.models:
            # Get compatible API providers for this model's provider
            compatible_providers = MODEL_TO_API_MAPPING.get(model.provider, [])

            # Find the first available API that can handle this model
            for api_provider in compatible_providers:
                api = next((api for api in self.model_apis if api.provider == api_provider), None)

                if api:
                    # Create a unique identifier for this model-API combination
                    endpoint_id = f"{model.model_name}_{api.provider}"

                    # Skip if already created
                    if endpoint_id in created_endpoints:
                        continue

                    # Create the endpoint
                    endpoint = AIModelEndpoint(
                        reference_number=endpoint_id,
                        ai_model=model,
                        api=api,
                    )

                    self.model_endpoints.append(endpoint)
                    created_endpoints.add(endpoint_id)

                    # Once we've created an endpoint for this model, move to the next model
                    break

    @classmethod
    def create_credentials_template(cls, output_file: str = "credentials.yaml") -> None:
        """
        Create a template credentials file with required fields for each provider.

        Args:
            output_file: Path to save the template file
        """
        # Start with a header comment
        header = (
            "# Dhenara AI Provider Credentials\n"
            "# Replace placeholder values with your actual API keys and remove unused items\n\n"
        )

        template = {}

        for provider_enum, config in PROVIDER_CONFIGS.items():
            provider_str = provider_enum.value
            template[provider_str] = {}

            # Add API key if required
            if config.api_key_required:
                template[provider_str]["api_key"] = f"<YOUR_{provider_str.upper()}_API_KEY>"

            # Add required credential fields
            for field_config in config.credentials_required_fields:
                field_name = field_config.field_name
                template[provider_str][field_name] = f"<YOUR_{provider_str.upper()}_{field_name.upper()}>"

            for field_config in config.config_required_fields:
                field_name = field_config.field_name
                template[provider_str][field_name] = f"<YOUR_{provider_str.upper()}_{field_name.upper()}>"

            # Do not add the optional fields directly to the template dictionary with the comment
            # Instead, we'll handle comments specially during YAML dumping

        # Write the template to a file
        ext = os.path.splitext(output_file)[1].lower()

        if ext == ".json":
            with open(output_file, "w") as f:
                json.dump(template, f, indent=2)
        else:
            # For YAML, we need to create the content manually to include comments properly
            yaml_content = header

            for provider_enum, config in PROVIDER_CONFIGS.items():
                provider_str = provider_enum.value
                yaml_content += f"{provider_str}:\n"

                # Add API key if required
                if config.api_key_required:
                    yaml_content += f"  api_key: <YOUR_{provider_str.upper()}_API_KEY>\n"

                # credentials
                if config.credentials_required_fields or config.credentials_optional_fields:
                    yaml_content += "  credentials:\n"

                # Add required credential fields
                for field_config in config.credentials_required_fields:
                    field_name = field_config.field_name
                    yaml_content += f"    {field_name}: <YOUR_{provider_str.upper()}_{field_name.upper()}>\n"

                # Add optional credential fields with comment
                for field_config in config.credentials_optional_fields:
                    field_name = field_config.field_name
                    yaml_content += f"    {field_name}: <YOUR_{provider_str.upper()}_{field_name.upper()}> # Optional\n"

                # config
                if config.config_required_fields or config.config_optional_fields:
                    yaml_content += "  config:\n"

                # Add required config fields
                for field_config in config.config_required_fields:
                    field_name = field_config.field_name
                    yaml_content += f"    {field_name}: <YOUR_{provider_str.upper()}_{field_name.upper()}>\n"

                # Add optional config fields with comment
                for field_config in config.config_optional_fields:
                    field_name = field_config.field_name
                    yaml_content += f"    {field_name}: <YOUR_{provider_str.upper()}_{field_name.upper()}> # Optional\n"

                # Add an extra newline between providers
                yaml_content += "\n"

            # Write the YAML content to file
            with open(output_file, "w") as f:
                f.write(yaml_content)

        logger.info(f"Created credentials template at {output_file}")
        logger.info("Edit this file with your API credentials before loading")
