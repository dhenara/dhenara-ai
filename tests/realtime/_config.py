from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pytest

from dhenara.ai.types import AIModelAPIProviderEnum, AIModelEndpoint, ResourceConfig

DEFAULT_PROVIDER_MODELS_MAP: dict[AIModelAPIProviderEnum, tuple[str, ...]] = {
    AIModelAPIProviderEnum.OPEN_AI: (
        "gpt-5-nano",
        # "gpt-5-mini",
        # "gpt-5",
    ),
    AIModelAPIProviderEnum.ANTHROPIC: (
        "claude-haiku-4-5",
        # "claude-sonnet-4-5",
    ),
    AIModelAPIProviderEnum.GOOGLE_VERTEX_AI: (
        "gemini-2.5-flash-lite",
        # "gemini-2.5-flash",
        # "gemini-2.5-pro",
    ),
}

ALL_PROVIDER_MODELS_MAP: dict[AIModelAPIProviderEnum, tuple[str, ...]] = {
    AIModelAPIProviderEnum.OPEN_AI: ("gpt-5-nano"),
    AIModelAPIProviderEnum.ANTHROPIC: ("claude-haiku-4-5"),
    AIModelAPIProviderEnum.GOOGLE_VERTEX_AI: ("gemini-2.5-flash-lite"),
    AIModelAPIProviderEnum.GOOGLE_AI: ("gemini-2.5-flash-lite"),
    AIModelAPIProviderEnum.AMAZON_BEDROCK: ("claude-haiku-4-5"),
    AIModelAPIProviderEnum.MICROSOFT_OPENAI: ("gpt-5-nano"),
}


@dataclass(frozen=True)
class EndpointMatch:
    provider: str | None = None
    model: str | None = None


def _normalise(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _parse_spec(spec: str) -> EndpointMatch:
    cleaned = spec.strip()
    if not cleaned:
        return EndpointMatch()
    if ":" in cleaned:
        provider, model = cleaned.split(":", 1)
        return EndpointMatch(provider=_normalise(provider), model=_normalise(model))
    return EndpointMatch(model=_normalise(cleaned))


def _endpoint_matches(endpoint: AIModelEndpoint, match: EndpointMatch) -> bool:
    provider_match = True
    model_match = True

    if match.provider:
        provider_value = getattr(endpoint.api.provider, "value", str(endpoint.api.provider)).lower()
        provider_name = getattr(endpoint.api.provider, "name", provider_value).lower()
        provider_match = match.provider.lower() in {provider_value, provider_name}

    if match.model:
        model_match = endpoint.ai_model.model_name.lower() == match.model.lower()

    return provider_match and model_match


def load_realtime_resource_config() -> ResourceConfig:
    config = ResourceConfig()
    credentials_path = None  # Load the default
    config.load_from_file(credentials_path, init_endpoints=True)
    return config


@pytest.fixture(scope="session")
def realtime_resource_config() -> ResourceConfig:
    try:
        config = load_realtime_resource_config()
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Realtime tests require credentials file: {exc}")
    if not config.model_endpoints:
        pytest.skip("Realtime tests require at least one configured endpoint")
    return config


def select_endpoint(
    resource_config: ResourceConfig,
    *,
    env_var: str | None = None,
    predicate: Callable[[AIModelEndpoint], bool] | None = None,
) -> AIModelEndpoint:
    candidates: Iterable[AIModelEndpoint] = resource_config.model_endpoints

    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            for chunk in env_value.split(","):
                match = _parse_spec(chunk)
                for endpoint in candidates:
                    if _endpoint_matches(endpoint, match):
                        return endpoint
            pytest.skip(f"No endpoint matches specification from {env_var}")

    if predicate:
        for endpoint in candidates:
            if predicate(endpoint):
                return endpoint

    try:
        return next(iter(candidates))
    except StopIteration as exc:  # pragma: no cover - defensive guard
        raise pytest.skip("No endpoints available") from exc


def select_provider_endpoint(
    resource_config: ResourceConfig,
    provider: AIModelAPIProviderEnum,
    *,
    preferred_models: Iterable[str] | None = None,
) -> AIModelEndpoint:
    candidates = [endpoint for endpoint in resource_config.model_endpoints if endpoint.api.provider == provider]
    if not candidates:
        pytest.skip(f"No endpoints configured for provider {provider.name}")

    priority_models = list(preferred_models or DEFAULT_PROVIDER_MODELS_MAP.get(provider, ()))

    for model_name in priority_models:
        for endpoint in candidates:
            if endpoint.ai_model.model_name.lower() == model_name.lower():
                return endpoint
        pytest.skip(f"No endpoint configured for provider {provider.name} with model {model_name}")

    return candidates[0]


def provider_model_cases() -> list[tuple[AIModelAPIProviderEnum, str]]:
    return [(provider, model_name) for provider, models in DEFAULT_PROVIDER_MODELS_MAP.items() for model_name in models]


def all_provider_model_cases() -> list[tuple[AIModelAPIProviderEnum, str]]:
    return [(provider, model_name) for provider, models in ALL_PROVIDER_MODELS_MAP.items() for model_name in models]
