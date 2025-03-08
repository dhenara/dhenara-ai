import json
from typing import Any

from pydantic import Field, field_validator, model_validator

from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.genai.ai_model import PROVIDER_CONFIGS
from dhenara.ai.types.shared.base import BaseModel


class AIModelAPI(BaseModel):
    """
    Pydantic model representing API credentials for AI model providers with built-in validation.
    """

    provider: AIModelAPIProviderEnum = Field(
        ...,
        description="The AI model provider",
    )
    api_key: str | None = Field(
        None,
        description="API key, if applicable ",
    )
    credentials: dict[str, Any] | None = Field(
        None,
        description="Dictionary of sensitive credentials otherthan API key.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-sensitive credential configuration/parameters.",
    )
    order: int = Field(
        0,
        description="Order for display purposes",
    )
    enabled: bool = Field(
        True,  # noqa: FBT003
        description="Whether this API is enabled",
    )
    reference_number: str | None = Field(
        None,
        description="Reference number. Should be unique if not None",
    )

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "AIModelAPI":
        """Validate provider-specific requirements"""
        provider_config = PROVIDER_CONFIGS.get(self.provider)
        if not provider_config:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Validate API key requirement
        if provider_config.api_key_required and not self.api_key:
            raise ValueError(f"API key is required for {self.provider}")

        # Validate required credentials
        if self.credentials:
            for field_config in provider_config.credentials_required_fields:
                value = self.credentials.get(field_config.field_name)
                if not value:
                    raise ValueError(f"Missing required credential: {field_config.field_name}")

                if field_config.is_json_field:
                    try:
                        if isinstance(value, dict):
                            parsed_value = value
                        else:
                            parsed_value = json.loads(value.strip())

                        if not isinstance(parsed_value, dict):
                            raise ValueError(f"JSON field {field_config.field_name} must be a dictionary")

                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in {field_config.field_name}: {e}")

        # Validate required config fields
        for field_config in provider_config.config_required_fields:
            if field_config.field_name not in self.config:
                raise ValueError(f"Missing required config: {field_config.field_name}")

        return self

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate API key format"""
        if v is not None and len(v) < 8:
            raise ValueError("API key must be at least 8 characters long")
        return v

    @field_validator("credentials")
    @classmethod
    def validate_credentials(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v

        # Create a copy of the dictionary
        validated = v.copy()

        # Handle JSON fields if needed
        for key, value in v.items():
            if isinstance(value, str) and key.endswith("_json"):  # TODO: Use get_credentials_fields_config_with_json()
                try:
                    # Remove leading/trailing whitespace and newlines
                    validated[key] = json.loads(value.strip())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {key}: {e}")

        return validated

    def get_provider_credentials(self) -> dict[str, Any]:
        """Get provider-specific credentials based on output mappings"""
        provider_config = PROVIDER_CONFIGS.get(self.provider)
        if not provider_config:
            raise ValueError(f"Unsupported provider: {self.provider}")

        result = {}
        for mapping in provider_config.output_mappings:
            value = None
            if mapping.source == "api_key":
                value = self.api_key
            elif mapping.source == "credentials" and mapping.source_key:
                value = self.credentials.get(mapping.source_key) if self.credentials else None
            elif mapping.source == "config" and mapping.source_key:
                value = self.config.get(mapping.source_key)

            if value is None and mapping.default_value is not None:
                value = mapping.default_value

            result[mapping.output_key] = value

        return result

    @classmethod
    def get_providers_form_config(cls) -> dict[str, dict]:
        """
        Get form configuration for all providers in a JSON-serializable format.

        Returns:
            Dictionary mapping provider names to their form configurations.
        """
        providers_config = {}

        for provider, config in PROVIDER_CONFIGS.items():
            credentials_fields = []
            config_fields = []

            # Process required credential fields
            for field in config.credentials_required_fields:
                credentials_fields.append(  # noqa: PERF401
                    {
                        "field_name": field.field_name,
                        "label": field.field_name.replace("_", " ").title(),
                        "type": "textarea" if field.field_name == "service_account_json" else "text",
                        "required": True,
                        "error_msg": field.error_msg,
                    }
                )

            # Process optional credential fields
            for field in config.credentials_optional_fields:
                credentials_fields.append(  # noqa: PERF401
                    {
                        "field_name": field.field_name,
                        "label": field.field_name.replace("_", " ").title(),
                        "type": "text",
                        "required": False,
                        "error_msg": field.error_msg,
                    }
                )

            # Process required config fields
            for field in config.config_required_fields:
                config_fields.append(  # noqa: PERF401
                    {
                        "field_name": field.field_name,
                        "label": field.field_name.replace("_", " ").title(),
                        "type": "text",
                        "required": True,
                        "error_msg": field.error_msg,
                    }
                )

            # Process optional config fields
            for field in config.config_optional_fields:
                config_fields.append(  # noqa: PERF401
                    {
                        "field_name": field.field_name,
                        "label": field.field_name.replace("_", " ").title(),
                        "type": "text",
                        "required": False,
                        "error_msg": field.error_msg,
                    }
                )

            providers_config[provider.value] = {
                "value": provider.value,
                "api_key_required": config.api_key_required,
                "credentials_fields": credentials_fields,
                "config_fields": config_fields,
            }

        return providers_config

    def get_masked_credentials(self) -> dict[str, Any]:
        """Get masked version of credentials for display"""
        masked = {}
        if self.api_key:
            masked["api_key"] = f"{self.api_key[:4]}...{self.api_key[-4:]}"
        if self.credentials:
            masked["credentials"] = {
                k: f"{str(v)[:4]}...{str(v)[-4:]}" if v else None for k, v in self.credentials.items()
            }
        if self.config:
            masked["config"] = self.config
        return masked

    @classmethod
    def validate_updates(
        cls,
        current_model: "AIModelAPI",
        api_key: str | None = None,
        credentials: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "AIModelAPI":
        """
        Validate updates to an existing model configuration while preserving all other fields.

        Args:
            current_model: Current AIModelAPI instance
            api_key: New API key to update
            credentials: New credentials to merge
            config: New config to merge

        Returns:
            AIModelAPI: New validated model instance with updates

        Raises:
            ValueError: If updates are invalid
        """
        # Start with current model's data
        current_data = current_model.model_dump()

        # Update only the fields that are provided
        updated_data = current_data.copy()

        if api_key is not None:
            updated_data["api_key"] = api_key

        if credentials is not None:
            updated_data["credentials"] = {
                **current_data.get("credentials", {}),
                **credentials,
            }

        if config is not None:
            updated_data["config"] = {
                **current_data.get("config", {}),
                **config,
            }

        # Create and validate new model with all fields
        try:
            updated_model = cls(**updated_data)
            return updated_model
        except Exception as e:
            raise ValueError(f"Invalid updates: {e}") from e

    @classmethod
    def validate_credentials_update(
        cls,
        current_model: "AIModelAPI",
        credentials: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate credentials update only."""
        return cls.validate_updates(
            current_model,
            credentials=credentials,
        ).credentials

    @classmethod
    def validate_config_update(
        cls,
        current_model: "AIModelAPI",
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate config update only."""
        return cls.validate_updates(
            current_model,
            config=config,
        ).config
