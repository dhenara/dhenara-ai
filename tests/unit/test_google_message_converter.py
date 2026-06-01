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


@pytest.mark.case_id("DAI-124")
def test_dai_124_google_message_conversion_preserves_grounding_annotations():
    msg = {
        "role": "model",
        "parts": [
            {"text": "Spain won Euro 2024."},
        ],
    }
    grounding_metadata = {
        "groundingChunks": [
            {"web": {"uri": "https://uefa.com", "title": "UEFA"}},
        ],
        "groundingSupports": [
            {
                "segment": {"startIndex": 0, "endIndex": 20, "text": "Spain won Euro 2024"},
                "groundingChunkIndices": [0],
            }
        ],
    }

    items = GoogleMessageConverter.provider_message_to_dai_content_items(
        message=msg,
        grounding_metadata=grounding_metadata,
    )

    assert len(items) == 1
    annotations = items[0].message_contents[0].annotations
    assert isinstance(annotations, list)
    assert annotations[0]["type"] == "url_citation"
    assert annotations[0]["url"] == "https://uefa.com"
