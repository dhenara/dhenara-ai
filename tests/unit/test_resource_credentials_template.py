# Provenance: Added to improve credentials template coverage (2026-01-20)

from __future__ import annotations

import pytest

from dhenara.ai.types.resource import ResourceConfig

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-050")
def test_dai_050_create_credentials_template(tmp_path):
    """GIVEN an output path
    WHEN ResourceConfig.create_credentials_template is called
    THEN a credentials template is written containing provider sections.
    """

    out = tmp_path / "creds.yaml"
    ResourceConfig.create_credentials_template(output_file=str(out))

    text = out.read_text(encoding="utf-8")
    assert "Dhenara AI Provider Credentials" in text
    # At least one well-known provider section should exist
    assert "openai:" in text
