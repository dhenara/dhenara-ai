from typing import Any

from pydantic import ConfigDict, Field, model_validator

from dhenara.ai.types.shared.base import BaseModel


class UsageChargeComponent(BaseModel):
    key: str = Field(
        ...,
        description="Stable component key for this portion of the total charge.",
    )
    cost: float = Field(
        ...,
        description="Cost contributed by this component before any markup.",
    )
    units: int | float | None = Field(
        default=None,
        description="Quantity used to compute this component.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit label for the recorded quantity.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional machine-readable metadata for this cost component.",
    )


class UsageCharge(BaseModel):
    cost: float = Field(
        ...,
        description="Cost",
    )
    charge: float | None = Field(
        default=None,
        description="Charge after considering internal expences and margins."
        " Will be  None if  `cost_multiplier_percentage` is not set in cost data.",
    )
    components: list[UsageChargeComponent] | None = Field(
        default=None,
        description="Optional breakdown of additive cost components that produced the total cost.",
    )


class HostedToolUsage(BaseModel):
    request_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Normalized hosted-tool invocation counts keyed by tool or aggregate name.",
    )
    token_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Normalized hosted-tool token counts keyed by token role or aggregate name.",
    )
    billing_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Provider-normalized hosted-tool billing counters keyed by priced unit name.",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional machine-readable hosted-tool details, including provider-specific usage payloads.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data):
        if isinstance(data, dict):
            normalized = dict(data)
            if "details" not in normalized and "metadata" in normalized:
                normalized["details"] = normalized["metadata"]
            return normalized
        return data

    @property
    def metadata(self) -> dict[str, Any] | None:
        return self.details

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        self.details = value


class ChatResponseUsage(BaseModel):
    """Token usage statistics for the chat completion"""

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None = Field(
        default=None,
        description="Number of tokens used for reasoning/thinking (o3-mini, o1, etc.). "
        "These are included in completion_tokens count.",
    )
    hosted_tool_usage: HostedToolUsage | None = Field(
        default=None,
        description="Standardized hosted-tool usage and billing data for cross-provider accounting.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data):
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        legacy_provider_usage = normalized.pop("provider_tool_usage", None)
        hosted_tool_usage = normalized.get("hosted_tool_usage")
        if legacy_provider_usage is not None:
            hosted_usage_data: dict[str, Any]
            if isinstance(hosted_tool_usage, HostedToolUsage):
                hosted_usage_data = hosted_tool_usage.model_dump()
            elif isinstance(hosted_tool_usage, dict):
                hosted_usage_data = dict(hosted_tool_usage)
            else:
                hosted_usage_data = {}

            details_raw = hosted_usage_data.get("details")
            details = dict(details_raw) if isinstance(details_raw, dict) else {}
            details.setdefault("provider_usage", legacy_provider_usage)
            hosted_usage_data["details"] = details
            normalized["hosted_tool_usage"] = hosted_usage_data

        return normalized

    @property
    def provider_tool_usage(self) -> dict[str, Any] | None:
        hosted_tool_usage = self.hosted_tool_usage
        if hosted_tool_usage is None or not isinstance(hosted_tool_usage.details, dict):
            return None
        provider_usage = hosted_tool_usage.details.get("provider_usage")
        return provider_usage if isinstance(provider_usage, dict) else None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
                "reasoning_tokens": 20,
                "hosted_tool_usage": {
                    "request_counts": {"web_search": 1, "total": 1},
                    "token_counts": {"prompt": 120},
                    "billing_counts": {"web_search": 1},
                },
            }
        },
    )


class ImageResponseUsage(BaseModel):
    """Usage information for image generation.
    Note that, for images, no usage data is received, so this class holds params required for usage/cost calculation"""

    number_of_images: int = Field(
        ...,
        description="Number of Images generated",
    )
    model: str = Field(
        default="",
        description="Model Name",
    )
    options: dict = Field(
        default_factory=dict,
        description="Options send to API",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "dall-e-3",
                "options": {
                    "size": "1024x1024",
                    "quality": "standard",
                },
            }
        },
    )
