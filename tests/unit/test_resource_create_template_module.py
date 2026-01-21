# Provenance: Added to cover resource create_template module wrapper (2026-01-21)

from __future__ import annotations

import importlib
import sys

import pytest

from dhenara.ai.types.resource._resource_config import ResourceConfig

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-051")
def test_dai_051_create_template_module_calls_generator(monkeypatch, capsys):
    """GIVEN the resource.create_template module
    WHEN it is imported
    THEN it calls ResourceConfig.create_credentials_template and swallows errors.
    """

    called = {"ok": 0}

    def _fake_create_credentials_template(*_args, **_kwargs):
        called["ok"] += 1

    monkeypatch.setattr(ResourceConfig, "create_credentials_template", staticmethod(_fake_create_credentials_template))

    mod_name = "dhenara.ai.types.resource.create_template"
    sys.modules.pop(mod_name, None)
    importlib.import_module(mod_name)

    assert called["ok"] == 1

    # Error path: exceptions are swallowed and printed
    def _boom(*_args, **_kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(ResourceConfig, "create_credentials_template", staticmethod(_boom))
    sys.modules.pop(mod_name, None)
    importlib.import_module(mod_name)

    out = capsys.readouterr().out
    assert "Error while creating file" in out
