import json
from pathlib import Path

import pytest

from dhenara.ai.types.genai.dhenara.request import Prompt, SystemInstruction
from dhenara.ai.utils.artifacts import ArtifactWriter

# NOTE: Test uses OpenAI provider class but does not perform real network calls;
# config will disable actual API usage by omitting key (provider logic should short-circuit or throw early).
# We only exercise artifact writing path via _capture_artifacts. If provider needs key, we skip real call.

APOS = chr(0x2019)  # RIGHT SINGLE QUOTATION MARK (U+2019)
EMOJI = chr(0x1F600)  # 😀
SMART_QUOTE = f"Here{APOS}s a line with a smart apostrophe {APOS} and emoji {EMOJI}"


def _write_dai_request(root: Path, prefix: str, data: dict):
    ArtifactWriter.write_json(artifact_root=root, filename="dai_request.json", data=data, prefix=f"{prefix}/dai")


@pytest.mark.component
@pytest.mark.case_id("DAI-001")
def test_dai_request_preserves_unicode(tmp_path):
    # Prepare data structure similar to serialized dai request
    # Build messages containing unicode
    messages = [
        Prompt.with_text(SMART_QUOTE),
    ]
    ql = chr(0x201D)  # ”
    # SystemInstruction expects "text" (str or PromptText); previous attempt passed invalid field names
    instructions = [SystemInstruction(text=f"System says: {ql} smart quotes too {ql}")]
    dai_request = {
        "prompt": None,
        "context": [],
        "instructions": [i.model_dump() for i in instructions],
        "messages": [m.model_dump() for m in messages],
        "config": {},
    }

    _write_dai_request(tmp_path, "test_run_001", dai_request)

    artifact_file = tmp_path / "test_run_001" / "dai" / "dai_request.json"
    assert artifact_file.exists(), "dai_request.json not written"

    raw_text = artifact_file.read_text(encoding="utf-8")
    # Ensure the smart apostrophe and emoji are not escaped as \uXXXX
    assert "\\u2019" not in raw_text, "Smart apostrophe was escaped"
    assert "\\uD83D" not in raw_text, "Emoji was escaped as surrogate pair"
    assert APOS in raw_text, "Smart apostrophe missing"
    assert EMOJI in raw_text, "Emoji missing"

    data = json.loads(raw_text)
    # Validate round-trip still contains unicode characters
    # Access the underlying template text where the user prompt resides
    first_msg_text = data["messages"][0]["text"]["template"]["text"]
    assert APOS in first_msg_text
    assert EMOJI in first_msg_text


@pytest.mark.component
@pytest.mark.case_id("DAI-002")
def test_jsonl_writer_preserves_unicode(tmp_path):
    ArtifactWriter.write_jsonl(
        artifact_root=tmp_path,
        filename="unicode.jsonl",
        rows=[{"text": SMART_QUOTE}, {"emoji": "😀"}],
        prefix="test_run_002/dai",
    )

    jf = tmp_path / "test_run_002" / "dai" / "unicode.jsonl"
    assert jf.exists()
    raw = jf.read_text(encoding="utf-8")
    assert "\\u2019" not in raw
    assert APOS in raw
    assert EMOJI in raw
