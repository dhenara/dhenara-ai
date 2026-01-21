# Provenance: Added to improve structured output parsing coverage (2026-01-20)

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig
from dhenara.ai.types.genai.dhenara.response._content_items._structured_output import ChatResponseStructuredOutput

pytestmark = [pytest.mark.unit]


class _Out(BaseModel):
    a: int = Field(..., strict=True)

    @staticmethod
    def schema_post_process_on_error(data):
        # Coerce a from string to int
        if isinstance(data, dict) and "a" in data and isinstance(data["a"], str):
            return {**data, "a": int(data["a"])}
        return data


@pytest.mark.case_id("DAI-048")
def test_dai_048_structured_output_parsing_and_salvage():
    """GIVEN structured output raw text with streaming artifacts and nested JSON strings
    WHEN parsed via ChatResponseStructuredOutput.parse_and_validate
    THEN it extracts the last JSON object and coerces nested JSON strings.
    """

    cfg = StructuredOutputConfig.from_model(_Out)

    # Multiple JSON objects concatenated (common streaming artifact): prefer the last one.
    raw = '{"a": 1}{"a": 2}'
    parsed, err, post = ChatResponseStructuredOutput.parse_and_validate(raw, cfg)
    assert err is None
    assert post is False
    assert parsed == {"a": 2}

    # Post-process fallback path
    cfg2 = StructuredOutputConfig.from_model(_Out)
    cfg2.allow_post_process_on_error = True
    parsed2, err2, post2 = ChatResponseStructuredOutput.parse_and_validate('{"a": "3"}', cfg2)
    assert err2 is None
    assert post2 is True
    assert parsed2 == {"a": 3}
