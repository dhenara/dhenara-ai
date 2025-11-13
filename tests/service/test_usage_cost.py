"""Service tests covering usage and cost toggles."""

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.config import settings
from dhenara.ai.types.genai import ChatResponseUsage

from ..helpers import FakeProvider


class _UsageProvider(FakeProvider):
    """Fake provider that always returns deterministic usage data."""

    def _get_usage_from_provider_response(self, response):  # type: ignore[override]
        return ChatResponseUsage(total_tokens=100, prompt_tokens=40, completion_tokens=60)


@pytest.mark.service
@pytest.mark.case_id("DAI-032")
def test_usage_cost_calculation_and_toggles(monkeypatch, text_endpoint, default_call_config):
    """
    GIVEN usage and cost tracking settings toggled
    WHEN generate() executes under each configuration
    THEN the response should reflect usage/charge presence in line with the toggles
    """

    def fake_provider(cls, model_endpoint, config, is_async):
        return _UsageProvider(model_endpoint=model_endpoint, config=config, is_async=is_async)

    monkeypatch.setattr(AIModelClientFactory, "create_provider_client", classmethod(fake_provider))

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=default_call_config,
        is_async=False,
    )

    # Ensure tracking enabled
    monkeypatch.setattr(settings, "ENABLE_USAGE_TRACKING", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_COST_TRACKING", True, raising=False)

    response = client.generate(prompt="usage on")
    usage = response.chat_response.usage if response.chat_response else None
    charge = response.chat_response.usage_charge if response.chat_response else None
    assert usage is not None and usage.prompt_tokens == 40
    assert charge is not None and charge.cost > 0

    # Disable tracking and ensure usage disappears
    monkeypatch.setattr(settings, "ENABLE_USAGE_TRACKING", False, raising=False)
    monkeypatch.setattr(settings, "ENABLE_COST_TRACKING", False, raising=False)

    response_disabled = client.generate(prompt="usage off")
    usage_disabled = response_disabled.chat_response.usage if response_disabled.chat_response else None
    charge_disabled = response_disabled.chat_response.usage_charge if response_disabled.chat_response else None
    assert usage_disabled is None
    assert charge_disabled is None
