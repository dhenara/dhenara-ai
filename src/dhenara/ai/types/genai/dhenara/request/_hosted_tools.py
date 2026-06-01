from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from dhenara.ai.types.genai.ai_model import AIModelProviderEnum


class HostedToolKind:
    WEB_SEARCH = "web_search"


class HostedToolUserLocation(BaseModel):
    type: Literal["approximate"] = "approximate"
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class WebSearchHostedTool(BaseModel):
    kind: Literal["hosted_tool"] = "hosted_tool"
    tool: Literal["web_search"] = HostedToolKind.WEB_SEARCH
    allowed_domains: list[str] | None = Field(default=None)
    blocked_domains: list[str] | None = Field(default=None)
    user_location: HostedToolUserLocation | None = Field(default=None)
    max_uses: int | None = Field(default=None)
    search_context_size: Literal["low", "medium", "high"] | None = Field(default=None)
    external_web_access: bool | None = Field(default=None)
    return_token_budget: Literal["default", "unlimited"] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


HostedToolDefinition = WebSearchHostedTool


class HostedToolProviderSpec(BaseModel):
    provider: AIModelProviderEnum
    provider_tool_type: str
    provider_tool_name: str | None = None


_HOSTED_TOOL_SUPPORT: dict[str, dict[AIModelProviderEnum, HostedToolProviderSpec]] = {
    HostedToolKind.WEB_SEARCH: {
        AIModelProviderEnum.OPEN_AI: HostedToolProviderSpec(
            provider=AIModelProviderEnum.OPEN_AI,
            provider_tool_type="web_search",
        ),
        AIModelProviderEnum.ANTHROPIC: HostedToolProviderSpec(
            provider=AIModelProviderEnum.ANTHROPIC,
            provider_tool_type="web_search_20260209",
            provider_tool_name="web_search",
        ),
        AIModelProviderEnum.GOOGLE_AI: HostedToolProviderSpec(
            provider=AIModelProviderEnum.GOOGLE_AI,
            provider_tool_type="google_search",
        ),
    }
}


def get_hosted_tool_support(
    tool_name: str,
) -> dict[AIModelProviderEnum, HostedToolProviderSpec]:
    return dict(_HOSTED_TOOL_SUPPORT.get(tool_name, {}))


def get_hosted_tool_provider_spec(
    tool: HostedToolDefinition,
    provider: AIModelProviderEnum,
) -> HostedToolProviderSpec:
    provider_specs = get_hosted_tool_support(tool.tool)
    provider_spec = provider_specs.get(provider)
    if provider_spec is None:
        raise ValueError(
            f"Hosted tool '{tool.tool}' is not supported for provider '{provider}'."
        )
    return provider_spec