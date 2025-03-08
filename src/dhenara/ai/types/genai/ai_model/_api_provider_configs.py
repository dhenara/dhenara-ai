from typing import Any, Literal

from pydantic import Field

from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.shared.base import BaseModel


class CredentialFieldConfig(BaseModel):
    """Configuration for a credential field"""

    field_name: str = Field(..., description="Name of the credential field")
    is_json_field: bool = Field(False, description="Whether the field contains JSON data")  # noqa: FBT003
    error_msg: str = Field(..., description="Error message for validation failures")


class CredentialOutputMapping(BaseModel):
    """Mapping configuration for credential output"""

    source: Literal["api_key", "credentials", "config"] = Field(
        ...,
        description="Source of the credential value",
    )
    source_key: str | None = Field(
        None,
        description="Key in the source dictionary (for credentials and config)",
    )
    output_key: str = Field(
        ...,
        description="Key name in the output credentials dictionary",
    )
    default_value: Any | None = Field(
        None,
        description="Default value if the source is not found",
    )


class ProviderCredentialsConfig(BaseModel):
    """Configuration for provider credentials"""

    api_key_required: bool = Field(..., description="Whether an API key is required")
    credentials_required_fields: list[CredentialFieldConfig] = Field(
        default_factory=list,
        description="Required credential fields",
    )
    credentials_optional_fields: list[CredentialFieldConfig] = Field(
        default_factory=list,
        description="Optional credential fields",
    )
    config_required_fields: list[CredentialFieldConfig] = Field(
        default_factory=list,
        description="Required configuration fields",
    )
    config_optional_fields: list[CredentialFieldConfig] = Field(
        default_factory=list,
        description="Optional configuration fields",
    )
    output_mappings: list[CredentialOutputMapping] = Field(
        default_factory=list,
        description="Mapping configuration for credential output",
    )

    def get_credentials_fields_config_with_json(self) -> list[CredentialFieldConfig]:
        all_credentials = self.credentials_required_fields + self.credentials_optional_fields
        return [field_config for field_config in all_credentials if field_config.is_json_field]


PROVIDER_CONFIGS: dict[AIModelAPIProviderEnum, ProviderCredentialsConfig] = {
    AIModelAPIProviderEnum.OPEN_AI: ProviderCredentialsConfig(
        api_key_required=True,
        output_mappings=[
            CredentialOutputMapping(
                source="api_key",
                output_key="api_key",
            ),
        ],
    ),
    AIModelAPIProviderEnum.GOOGLE_AI: ProviderCredentialsConfig(
        api_key_required=True,
        output_mappings=[
            CredentialOutputMapping(
                source="api_key",
                output_key="api_key",
            ),
        ],
    ),
    AIModelAPIProviderEnum.ANTHROPIC: ProviderCredentialsConfig(
        api_key_required=True,
        output_mappings=[
            CredentialOutputMapping(
                source="api_key",
                output_key="api_key",
            ),
        ],
    ),
    AIModelAPIProviderEnum.DEEPSEEK: ProviderCredentialsConfig(
        api_key_required=True,
        output_mappings=[
            CredentialOutputMapping(
                source="api_key",
                output_key="api_key",
            ),
        ],
    ),
    AIModelAPIProviderEnum.GOOGLE_VERTEX_AI: ProviderCredentialsConfig(
        api_key_required=False,
        credentials_required_fields=[
            CredentialFieldConfig(
                field_name="service_account_json",
                is_json_field=True,
                error_msg="Service account JSON must be a valid dictionary with project_id",
            ),
        ],
        config_required_fields=[
            CredentialFieldConfig(
                field_name="project_id",
                error_msg="Project ID must be a non-empty string",
            ),
            CredentialFieldConfig(
                field_name="location",
                error_msg="Location must be a non-empty string",
            ),
        ],
        output_mappings=[
            CredentialOutputMapping(
                source="credentials",
                source_key="service_account_json",
                output_key="service_account_json",
            ),
            CredentialOutputMapping(
                source="config",
                source_key="project_id",
                output_key="project_id",
            ),
            CredentialOutputMapping(
                source="config",
                source_key="location",
                output_key="location",
                default_value="us-central1",
            ),
        ],
    ),
    AIModelAPIProviderEnum.AMAZON_BEDROCK: ProviderCredentialsConfig(
        api_key_required=False,
        credentials_required_fields=[
            CredentialFieldConfig(
                field_name="access_key_id",
                error_msg="Invalid AWS access key ID format. Should start with 'AKIA'",
            ),
            CredentialFieldConfig(
                field_name="secret_access_key",
                error_msg="Invalid AWS secret access key format. Should be at least 40 characters long",
            ),
        ],
        config_required_fields=[
            CredentialFieldConfig(
                field_name="region",
                error_msg="Region must be a non-empty string",
            ),
        ],
        output_mappings=[
            CredentialOutputMapping(
                source="credentials",
                source_key="access_key_id",
                output_key="aws_access_key",
            ),
            CredentialOutputMapping(
                source="credentials",
                source_key="secret_access_key",
                output_key="aws_secret_key",
            ),
            CredentialOutputMapping(
                source="config",
                source_key="region",
                output_key="aws_region",
            ),
        ],
    ),
    AIModelAPIProviderEnum.MICROSOFT_OPENAI: ProviderCredentialsConfig(
        api_key_required=True,
        credentials_required_fields=[],
        credentials_optional_fields=[],
        config_required_fields=[
            CredentialFieldConfig(
                field_name="endpoint",
                error_msg="endpoint must be a non-empty string",
            ),
        ],
        config_optional_fields=[
            CredentialFieldConfig(
                field_name="api_version",
                error_msg="api_version must be a non-empty string",
            ),
        ],
        output_mappings=[
            CredentialOutputMapping(
                source="api_key",
                output_key="api_key",
            ),
            CredentialOutputMapping(
                source="config",
                source_key="endpoint",
                output_key="azure_endpoint",
            ),
            CredentialOutputMapping(
                source="config",
                source_key="api_version",
                output_key="api_version",
                default_value="2024-10-21",
            ),
        ],
    ),
    AIModelAPIProviderEnum.MICROSOFT_AZURE_AI: ProviderCredentialsConfig(
        api_key_required=True,
        credentials_required_fields=[],
        credentials_optional_fields=[],
        config_required_fields=[
            CredentialFieldConfig(
                field_name="endpoint",
                error_msg="endpoint must be a non-empty string",
            ),
        ],
        config_optional_fields=[],
        output_mappings=[
            CredentialOutputMapping(
                source="api_key",
                output_key="api_key",
            ),
            CredentialOutputMapping(
                source="config",
                source_key="endpoint",
                output_key="azure_endpoint",
            ),
        ],
    ),
}
