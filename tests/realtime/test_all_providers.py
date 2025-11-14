from __future__ import annotations

import pytest

from ._helpers import (
    run_structured_output_all_providers,
)

pytestmark = pytest.mark.realtime


def test_realtime_structured_output_all_providers(realtime_resource_config):
    """
    GIVEN the realtime resource configuration
    WHEN we sweep structured output across providers
    THEN every visited provider returns structured content
    """

    results = run_structured_output_all_providers(realtime_resource_config)
    assert results
