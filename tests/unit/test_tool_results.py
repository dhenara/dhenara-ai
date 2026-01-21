# Provenance: Added to improve tool result model coverage (2026-01-21)

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dhenara.ai.types.genai.dhenara.request.data._tool_result import ToolCallResult, ToolCallResultsMessage

pytestmark = [pytest.mark.unit]


class _NonJson:
    def __str__(self) -> str:  # pragma: no cover
        return "NONJSON"


@pytest.mark.case_id("DAI-067")
def test_dai_067_tool_call_result_serialization_and_validation():
    """GIVEN ToolCallResult and ToolCallResultsMessage models
    WHEN outputs are strings, JSON objects, or non-JSON values
    THEN serialization helpers and validators behave predictably.
    """

    r_text = ToolCallResult(call_id="c1", output="hello")
    assert r_text.as_text() == "hello"
    assert r_text.as_json() == {"result": "hello"}

    r_obj = ToolCallResult(call_id="c2", output={"a": 1})
    assert r_obj.as_json() == {"a": 1}
    assert '"a"' in r_obj.as_text()

    r_nonjson = ToolCallResult(call_id="c3", output=_NonJson())
    assert "NONJSON" in r_nonjson.as_text()
    assert r_nonjson.as_json() == {"result": r_nonjson.output}

    msg = ToolCallResultsMessage(results=[r_text, r_obj])
    assert msg.as_list()[0].call_id == "c1"

    with pytest.raises(ValidationError, match=r"must contain at least one"):
        ToolCallResultsMessage(results=[])
