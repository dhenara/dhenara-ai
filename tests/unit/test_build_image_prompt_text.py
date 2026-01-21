# Provenance: Added to improve image prompt text builder coverage (2026-01-21)

from __future__ import annotations

import pytest

from dhenara.ai.providers.common.message_text import build_image_prompt_text
from dhenara.ai.types.genai.dhenara.request import Prompt

pytestmark = [pytest.mark.unit]


class _DummyFormatter:
    def join_instructions(self, instructions):
        return " | ".join(str(i) for i in instructions)


@pytest.mark.case_id("DAI-052")
def test_dai_052_build_image_prompt_text_messages_and_context_paths():
    """GIVEN image prompt inputs in messages format and legacy prompt/context format
    WHEN build_image_prompt_text is called
    THEN it produces a single non-empty prompt string with instructions included.
    """

    msg_prompt = Prompt.with_text("draw a cat")
    out1 = build_image_prompt_text(
        prompt=None,
        context=None,
        instructions=["system: be concise"],
        messages=[msg_prompt],
        formatter=_DummyFormatter(),
    )
    assert "system: be concise" in out1
    assert "draw a cat" in out1

    out2 = build_image_prompt_text(
        prompt="draw a dog",
        context=[{"content": "ctx1"}, "ctx2"],
        instructions={"content": "system: follow rules"},
        messages=None,
        formatter=None,
    )
    assert "follow rules" in out2
    assert "ctx1" in out2
    assert "ctx2" in out2
    assert "draw a dog" in out2

    with pytest.raises(ValueError, match=r"Image generation requires"):
        build_image_prompt_text(prompt=None, context=None, instructions=None, messages=None)
