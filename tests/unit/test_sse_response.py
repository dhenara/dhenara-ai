# Provenance: Added to improve SSE parse/format coverage (2026-01-20)

from __future__ import annotations

import pytest
from pydantic import BaseModel

from dhenara.ai.types.shared.api import (
    SSEErrorCode,
    SSEErrorResponse,
    SSEEventType,
    SSEResponse,
)

pytestmark = [pytest.mark.unit]


class _TypedPayload(BaseModel):
    x: int


@pytest.mark.case_id("DAI-039")
def test_dai_039_to_sse_format_dict_and_multiline():
    """GIVEN an SSEResponse with a dict payload containing newlines
    WHEN converted to SSE wire format
    THEN it emits event/id and splits data across multiple data lines.
    """

    # Dict payload will be json-dumped, so embedded newlines become escaped ("\\n")
    # and therefore do not produce multiple SSE `data:` lines.
    resp_dict = SSEResponse(event=SSEEventType.TOKEN_STREAM, data={"text": "a\nb"}, id="evt1")
    sse_dict = resp_dict.to_sse_format()

    assert "event: token_stream" in sse_dict
    assert "id: evt1" in sse_dict
    assert sse_dict.count("data: ") == 1

    # String payload with actual newlines should become multiple `data:` lines.
    resp_text = SSEResponse(event=SSEEventType.TOKEN_STREAM, data="a\nb", id="evt1")
    sse_text = resp_text.to_sse_format()
    assert sse_text.count("data: ") == 2
    assert sse_text.endswith("\n\n")


@pytest.mark.case_id("DAI-040")
def test_dai_040_parse_sse_typed_and_error_paths():
    """GIVEN valid and invalid SSE strings
    WHEN parsed via SSEResponse.parse_sse
    THEN typed payloads decode, and invalid shapes return SSEErrorResponse.
    """

    ok = 'event: token_stream\nid: 1\ndata: {"x": 7}\n\n'
    parsed = SSEResponse.parse_sse(ok, data_type=_TypedPayload)
    assert isinstance(parsed, SSEResponse)
    assert parsed.event == SSEEventType.TOKEN_STREAM
    assert parsed.data.x == 7

    bad_event = "event: not-a-real-event\nid: 1\ndata: {}\n\n"
    parsed_bad_event = SSEResponse.parse_sse(bad_event, data_type=_TypedPayload)
    assert isinstance(parsed_bad_event, SSEErrorResponse)
    assert parsed_bad_event.data.error_code == SSEErrorCode.client_decode_error

    bad_json = "event: token_stream\nid: 1\ndata: {not json}\n\n"
    parsed_bad_json = SSEResponse.parse_sse(bad_json)
    assert isinstance(parsed_bad_json, SSEErrorResponse)
    assert parsed_bad_json.data.error_code == SSEErrorCode.client_decode_error

    missing_data = "event: token_stream\nid: 1\n\n"
    parsed_missing = SSEResponse.parse_sse(missing_data)
    assert isinstance(parsed_missing, SSEErrorResponse)
    assert parsed_missing.data.error_code == SSEErrorCode.client_decode_error
