# Provenance: Added to improve Settings config loading coverage (2026-01-21)

from __future__ import annotations

import importlib.util as importlib_util

import pytest

from dhenara.ai.config import Settings

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-063")
def test_dai_063_loads_user_settings_from_cwd(tmp_path, monkeypatch):
    """GIVEN a dhenara_config.py present in the current working directory
    WHEN Settings is initialized
    THEN it loads upper-case settings from the user file.
    """

    (tmp_path / "dhenara_config.py").write_text(
        "MY_SETTING = 'hello'\nnot_exported = 123\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

    s = Settings()

    assert s.MY_SETTING == "hello"
    assert "MY_SETTING" in s.configured_settings


@pytest.mark.case_id("DAI-064")
def test_dai_064_invalid_user_settings_spec_logs_error(tmp_path, monkeypatch, caplog):
    """GIVEN a dhenara_config.py exists but cannot be imported (invalid spec/loader)
    WHEN Settings is initialized
    THEN it logs an error and continues without raising.
    """

    (tmp_path / "dhenara_config.py").write_text("MY_SETTING = 'hello'\n", encoding="utf-8")
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

    def _bad_spec_from_file_location(*args, **kwargs):
        return None

    monkeypatch.setattr(importlib_util, "spec_from_file_location", _bad_spec_from_file_location)

    with caplog.at_level("ERROR"):
        s = Settings()

    assert "Unable to load settings" in caplog.text
    with pytest.raises(AttributeError, match=r"Setting 'MY_SETTING' not found"):
        _ = s.MY_SETTING
