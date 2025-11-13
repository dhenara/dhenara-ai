"""Unit tests for ArtifactWriter utilities.

Tests cover text and JSONL writing with unicode preservation.
"""

import json
from pathlib import Path

import pytest

from dhenara.ai.utils.artifacts import ArtifactWriter

APOS = chr(0x2019)  # RIGHT SINGLE QUOTATION MARK (U+2019)
EMOJI = chr(0x1F600)  # 😀


@pytest.mark.unit
@pytest.mark.case_id("DAI-011")
def test_write_text_and_append_jsonl_preserves_unicode(tmp_path):
    """
    GIVEN unicode content with smart quotes and emojis
    WHEN write_text is called
    THEN the text file should preserve unicode characters without escaping

    WHEN append_jsonl is called with unicode data
    THEN the JSONL file should preserve unicode characters
    """
    # Test write_text with unicode
    unicode_text = f"Here{APOS}s a line with smart quote and emoji {EMOJI}"

    ArtifactWriter.write_text(
        artifact_root=tmp_path,
        filename="unicode_test.txt",
        content=unicode_text,
        prefix="test_artifacts",
    )

    text_file = tmp_path / "test_artifacts" / "unicode_test.txt"
    assert text_file.exists()

    # Read back and verify unicode is preserved
    content = text_file.read_text(encoding="utf-8")
    assert APOS in content
    assert EMOJI in content
    assert "\\u2019" not in content  # Should not be escaped
    assert unicode_text == content

    # Test append_jsonl with unicode
    unicode_rows = [
        {"text": f"Smart quote{APOS} here"},
        {"emoji": EMOJI},
        {"mixed": f"{APOS} and {EMOJI} together"},
    ]

    # First write creates file
    ArtifactWriter.write_jsonl(
        artifact_root=tmp_path,
        filename="unicode_append.jsonl",
        rows=[unicode_rows[0]],
        prefix="test_artifacts",
    )

    # Then append
    ArtifactWriter.append_jsonl(
        artifact_root=tmp_path,
        filename="unicode_append.jsonl",
        rows=unicode_rows[1:],
        prefix="test_artifacts",
    )

    jsonl_file = tmp_path / "test_artifacts" / "unicode_append.jsonl"
    assert jsonl_file.exists()

    # Read and verify
    raw_content = jsonl_file.read_text(encoding="utf-8")
    assert APOS in raw_content
    assert EMOJI in raw_content
    assert "\\u2019" not in raw_content
    assert "\\uD83D" not in raw_content  # Emoji should not be escaped as surrogate

    # Parse lines and verify data integrity
    lines = raw_content.strip().split("\n")
    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert APOS in parsed[0]["text"]
    assert parsed[1]["emoji"] == EMOJI
    assert APOS in parsed[2]["mixed"]
    assert EMOJI in parsed[2]["mixed"]

    # Test write_jsonl (not append) also preserves unicode
    ArtifactWriter.write_jsonl(
        artifact_root=tmp_path,
        filename="unicode_write.jsonl",
        rows=[{"value": f"Test{APOS}{EMOJI}"}],
    )

    write_file = tmp_path / "unicode_write.jsonl"
    write_content = write_file.read_text(encoding="utf-8")
    assert APOS in write_content
    assert EMOJI in write_content
