"""Unit tests for AIModelEndpoint cost calculation.

Tests cover usage charge calculation from cost data.
"""

import pytest

from dhenara.ai.types.genai.ai_model import (
    AIModelAPI,
    AIModelAPIProviderEnum,
    AIModelEndpoint,
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    ChatModelCostData,
    ChatModelSettings,
    ChatResponseUsage,
    FoundationModel,
    HostedToolCostRule,
    HostedToolUsage,
)


@pytest.mark.unit
@pytest.mark.case_id("DAI-010")
def test_calculate_usage_charge_from_cost_data():
    """
    GIVEN an AIModelEndpoint with ChatModelCostData
    WHEN calculate_usage_charge is called with ChatResponseUsage
    THEN it should correctly compute the cost based on token counts and rates
    AND apply cost_multiplier_percentage if provided
    """
    # Create test model and API
    model = FoundationModel(
        model_name="test-model",
        display_name="Test Model",
        provider=AIModelProviderEnum.OPEN_AI,
        functional_type=AIModelFunctionalTypeEnum.TEXT_GENERATION,
        settings=ChatModelSettings(
            max_input_tokens=100000,
            max_output_tokens=4096,
        ),
        valid_options={},
        cost_data=ChatModelCostData(
            input_token_cost_per_million=5.0,
            output_token_cost_per_million=15.0,
        ),
    )

    api = AIModelAPI(
        provider=AIModelAPIProviderEnum.OPEN_AI,
        api_key="test-key",
    )

    # Test 1: Basic cost calculation without multiplier
    # Don't set cost_data on endpoint - let it use the model's cost_data
    endpoint = AIModelEndpoint(
        api=api,
        ai_model=model,
    )

    usage = ChatResponseUsage(
        total_tokens=1500,
        prompt_tokens=1000,  # 1000 * ($5/1M) = $0.005
        completion_tokens=500,  # 500 * ($15/1M) = $0.0075
    )

    charge = endpoint.calculate_usage_charge(usage)

    # Expected cost: (1000 * 5 + 500 * 15) / 1,000,000 = 0.0125
    assert charge.cost == 0.0125
    assert charge.charge is None  # No multiplier, so charge should be None

    # Test 2: With cost multiplier (e.g., 20% markup)
    cost_data_with_multiplier = ChatModelCostData(
        input_token_cost_per_million=5.0,
        output_token_cost_per_million=15.0,
        cost_multiplier_percentage=20.0,  # 20% markup
    )

    endpoint_with_multiplier = AIModelEndpoint(
        api=api,
        ai_model=model,
        cost_data=cost_data_with_multiplier,
    )

    charge_with_multiplier = endpoint_with_multiplier.calculate_usage_charge(usage)

    # Cost should be same
    assert charge_with_multiplier.cost == 0.0125
    # Charge should be cost * 1.20 = 0.015
    assert charge_with_multiplier.charge == 0.015

    # Test 3: Different token counts
    usage2 = ChatResponseUsage(
        total_tokens=2000,
        prompt_tokens=500,
        completion_tokens=1500,
    )

    charge2 = endpoint.calculate_usage_charge(usage2)
    # (500 * 5 + 1500 * 15) / 1,000,000 = 0.025
    assert charge2.cost == 0.025

    # Test 4: Zero tokens
    usage_zero = ChatResponseUsage(
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
    )

    charge_zero = endpoint.calculate_usage_charge(usage_zero)
    assert charge_zero.cost == 0.0

    # Test 5: Hosted/provider-side tool costs are additive to token pricing
    endpoint_with_hosted_tool_costs = AIModelEndpoint(
        api=api,
        ai_model=model,
        cost_data=ChatModelCostData(
            input_token_cost_per_million=5.0,
            output_token_cost_per_million=15.0,
            hosted_tool_cost_rules=[
                HostedToolCostRule(
                    key="hosted_tool:web_search",
                    usage_bucket="billing_counts",
                    usage_key="web_search",
                    flat_cost_per_unit=0.025,
                    unit="request",
                ),
                HostedToolCostRule(
                    key="hosted_tool:prompt_tokens",
                    usage_bucket="token_counts",
                    usage_key="prompt",
                    cost_per_million=20.0,
                    unit="token",
                ),
            ],
        ),
    )

    usage_with_hosted_tools = ChatResponseUsage(
        total_tokens=2600,
        prompt_tokens=1000,
        completion_tokens=500,
        hosted_tool_usage=HostedToolUsage(
            request_counts={"web_search": 2, "total": 2},
            token_counts={"prompt": 1000},
            billing_counts={"web_search": 2},
        ),
    )

    charge_with_hosted_tools = endpoint_with_hosted_tool_costs.calculate_usage_charge(usage_with_hosted_tools)
    # Token cost: 0.0125, web search: 0.05, hosted prompt tokens: 0.02 => 0.0825 total.
    assert charge_with_hosted_tools.cost == 0.0825
    assert charge_with_hosted_tools.components is not None
    assert {component.key for component in charge_with_hosted_tools.components} >= {
        "input_tokens",
        "output_tokens",
        "hosted_tool:web_search",
        "hosted_tool:prompt_tokens",
    }
