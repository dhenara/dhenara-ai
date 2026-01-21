# Provenance: Added to improve Google converter coverage (2026-01-20)

from __future__ import annotations

import pytest

from dhenara.ai.providers.google.message_converter import GoogleMessageConverter

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-046")
def test_dai_046_google_message_conversion_text_and_image():
    """GIVEN a Google message-like payload with parts
    WHEN converted to Dhenara content items
    THEN text parts become text items and unknown/inline data becomes generic items.
    """

    msg = {
        "role": "model",
        "parts": [
            {"text": "hello"},
            {"inline_data": {"mime_type": "image/png", "data": "AAAA"}},
        ],
    }

    items = GoogleMessageConverter.provider_message_to_dai_content_items(message=msg)
    assert len(items) == 2
    assert items[0].get_text() == "hello"
    # Second item is generic (inline_data)
    assert items[1].metadata
    assert "inline_data" in items[1].metadata
