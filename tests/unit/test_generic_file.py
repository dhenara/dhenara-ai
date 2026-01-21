# Provenance: Added to improve shared file model coverage (2026-01-20)

from __future__ import annotations

import pytest

from dhenara.ai.types.shared.file import GenericFile

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-049")
def test_dai_049_generic_file_roundtrip():
    """GIVEN a GenericFile
    WHEN serialized and validated again
    THEN the name and optional metadata are preserved.
    """

    gf = GenericFile(name="foo.txt", metadata=None)
    dumped = gf.model_dump()
    gf2 = GenericFile.model_validate(dumped)

    assert gf2.name == "foo.txt"
    assert gf2.metadata is None
