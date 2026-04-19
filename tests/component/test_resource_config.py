# Provenance: Added to improve resource config coverage (2026-01-20)

from __future__ import annotations

import json
import re

import pytest
import yaml

from dhenara.ai.types.genai.foundation_models import ALL_FOUNDATION_MODELS
from dhenara.ai.types.resource import ResourceConfig

pytestmark = [pytest.mark.component]


def _first_openai_foundation_model():
    return next(m for m in ALL_FOUNDATION_MODELS if getattr(m, "provider", None) == "open_ai")


@pytest.mark.case_id("DAI-042")
def test_dai_042_load_credentials_yaml_and_json(tmp_path):
    """GIVEN YAML and JSON credentials files
    WHEN ResourceConfig.load_from_file is called
    THEN APIs are initialized and validation-required fields are enforced.
    """

    yaml_path = tmp_path / "creds.yaml"
    yaml_path.write_text(
        "openai:\n  api_key: sk-testkey123\n  credentials: {}\n  config: {}\n",
        encoding="utf-8",
    )

    rc = ResourceConfig()
    rc.load_from_file(credentials_file=str(yaml_path), init_endpoints=False)
    assert rc.model_apis
    assert rc.model_apis[0].provider == "openai"

    json_path = tmp_path / "creds.json"
    json_path.write_text(
        json.dumps({"openai": {"api_key": "sk-testkey123", "credentials": {}, "config": {}}}),
        encoding="utf-8",
    )

    rc2 = ResourceConfig()
    rc2.load_from_file(credentials_file=str(json_path), init_endpoints=False)
    assert rc2.model_apis
    assert rc2.model_apis[0].provider == "openai"


@pytest.mark.case_id("DAI-043")
def test_dai_043_initialize_endpoints_dedup(tmp_path):
    """GIVEN a ResourceConfig with a single OpenAI API and a single OpenAI model
    WHEN init_endpoints is enabled
    THEN exactly one endpoint is created for the model/api pair (no duplicates).
    """

    creds = tmp_path / "creds.yaml"
    creds.write_text(
        "openai:\n  api_key: sk-testkey123\n  credentials: {}\n  config: {}\n",
        encoding="utf-8",
    )

    fm = _first_openai_foundation_model()
    model = fm.create_instance() if hasattr(fm, "create_instance") else fm

    rc = ResourceConfig()
    # Duplicate the same model in one init pass; the internal created_endpoints set should dedup.
    rc.load_from_file(credentials_file=str(creds), models=[model, model], init_endpoints=True)

    assert len(rc.model_apis) == 1
    assert len(rc.model_endpoints) == 1

    ep = rc.model_endpoints[0]
    assert ep.ai_model.model_name == model.model_name
    assert ep.api.provider == "openai"


@pytest.mark.case_id("DAI-044")
def test_dai_044_load_credentials_defaults_to_secret_config_dir(tmp_path, monkeypatch):
    """GIVEN DAI_SECRET_CONFIG_DIR points at a host secret directory
    WHEN ResourceConfig.load_from_file is called without an explicit credentials path
    THEN it loads dai_credentials.yaml from that directory.
    """

    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    creds = secret_dir / "dai_credentials.yaml"
    creds.write_text(
        "openai:\n  api_key: sk-testkey123\n  credentials: {}\n  config: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DAI_SECRET_CONFIG_DIR", str(secret_dir))

    rc = ResourceConfig()
    rc.load_from_file(credentials_file=None, init_endpoints=False)

    assert rc.model_apis
    assert rc.model_apis[0].provider == "openai"


@pytest.mark.case_id("DAI-045")
def test_dai_045_missing_default_credentials_file_raises(tmp_path, monkeypatch):
    """GIVEN DAI_SECRET_CONFIG_DIR points at a directory without dai_credentials.yaml
    WHEN ResourceConfig.load_from_file is called without an explicit credentials path
    THEN it raises FileNotFoundError instead of silently falling back.
    """

    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    monkeypatch.setenv("DAI_SECRET_CONFIG_DIR", str(secret_dir))

    rc = ResourceConfig()
    with pytest.raises(FileNotFoundError, match=re.escape("dai_credentials.yaml")):
        rc.load_from_file(credentials_file=None, init_endpoints=False)


@pytest.mark.case_id("DAI-046")
def test_dai_046_invalid_credentials_yaml_raises(tmp_path):
    """GIVEN a credentials file with malformed YAML
    WHEN ResourceConfig.load_from_file is called
    THEN the package surfaces the parse error.
    """

    creds = tmp_path / "creds.yaml"
    creds.write_text("openai: [unterminated\n", encoding="utf-8")

    rc = ResourceConfig()
    with pytest.raises(yaml.YAMLError):
        rc.load_from_file(credentials_file=str(creds), init_endpoints=False)


@pytest.mark.case_id("DAI-047")
def test_dai_047_standalone_package_ignores_dvi_secret_env(tmp_path, monkeypatch):
    """GIVEN both DAI and DVI secret-dir env vars are present
    WHEN the standalone package resolves its default credentials file
    THEN it uses the DAI-scoped contract instead of the DVI-scoped one.
    """

    dai_secret_dir = tmp_path / "dai-secrets"
    dai_secret_dir.mkdir()
    dvi_secret_dir = tmp_path / "dvi-secrets"
    dvi_secret_dir.mkdir()
    (dvi_secret_dir / "dai_credentials.yaml").write_text(
        "openai:\n  api_key: sk-testkey123\n  credentials: {}\n  config: {}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("DAI_SECRET_CONFIG_DIR", str(dai_secret_dir))
    monkeypatch.setenv("DVI_SECRET_CONFIG_DIR", str(dvi_secret_dir))

    rc = ResourceConfig()
    with pytest.raises(FileNotFoundError, match=re.escape(str(dai_secret_dir / "dai_credentials.yaml"))):
        rc.load_from_file(credentials_file=None, init_endpoints=False)
