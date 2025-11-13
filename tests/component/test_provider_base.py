"""Component tests for provider base utilities and artifact capture."""

import json
from pathlib import Path

import pytest

from dhenara.ai.ai_client.ai_client import AIModelClient
from dhenara.ai.ai_client.factory import AIModelClientFactory
from dhenara.ai.types import ExternalApiCallStatusEnum
from dhenara.ai.types.genai.dhenara.request import ArtifactConfig, Prompt

from ..helpers import FakeProvider, make_fake_provider


@pytest.mark.component
@pytest.mark.case_id("DAI-016")
def test_format_inputs_mutual_exclusion_and_empty_errors(text_endpoint, default_call_config):
    """
    GIVEN a FakeProvider that relies on the base format_inputs implementation
    WHEN both prompt/context and messages are supplied together or both absent
    THEN format_inputs should return None signalling validation failure
    AND valid prompt-only input should succeed with formatted payloads
    """

    provider = FakeProvider(model_endpoint=text_endpoint, config=default_call_config, is_async=False)

    # Mixed prompt + messages should fail validation
    mixed = provider.format_inputs(prompt="hi", messages=[Prompt.with_text("there")])
    assert mixed is None

    # Absence of both prompt and messages should also fail
    missing = provider.format_inputs(prompt=None, context=None, messages=None)
    assert missing is None

    # Prompt-only path should succeed and include formatted prompt
    valid = provider.format_inputs(prompt="hello", context=["ctx"], messages=None)
    assert valid["prompt"]["text"] == "hello"
    assert valid["context"][0]["text"] == "ctx"


@pytest.mark.component
@pytest.mark.case_id("DAI-017")
def test_capture_artifacts_respects_flags(monkeypatch, text_endpoint, default_call_config, tmp_path):
    """
    GIVEN artifact capture enabled with selective stage flags
    WHEN generate() executes
    THEN only the enabled artifact files should be written beneath the dai prefix
    """

    artifact_config = ArtifactConfig(
        enabled=True,
        artifact_root=tmp_path,
        prefix="case_dai_017",
        capture_dhenara_request=True,
        capture_provider_request=False,
        capture_provider_response=True,
        capture_dhenara_response=False,
    )

    config = default_call_config.model_copy(update={"artifact_config": artifact_config})

    monkeypatch.setattr(
        AIModelClientFactory,
        "create_provider_client",
        classmethod(make_fake_provider()),
    )

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=config,
        is_async=False,
    )

    response = client.generate(prompt="collect artefacts")

    assert response.status is not None
    assert response.status.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS

    artifact_dir = Path(tmp_path) / "case_dai_017" / "dai"
    assert (artifact_dir / "dai_request.json").exists()
    assert not (artifact_dir / "dai_provider_request.json").exists()
    assert (artifact_dir / "dai_provider_response.json").exists()
    assert not (artifact_dir / "dai_response.json").exists()


@pytest.mark.component
@pytest.mark.case_id("DAI-018")
def test_python_log_capture_start_stop_and_jsonl(monkeypatch, text_endpoint, default_call_config, tmp_path):
    """
    GIVEN python log capture enabled for a call
    WHEN generate() completes
    THEN dai_python_logs.jsonl should exist with recorded log entries
    AND the provider log capture state should be reset afterwards
    """

    artifact_config = ArtifactConfig(
        enabled=True,
        artifact_root=tmp_path,
        prefix="case_dai_018",
        enable_python_logs=True,
        python_log_level="INFO",
    )

    config = default_call_config.model_copy(update={"artifact_config": artifact_config})

    monkeypatch.setattr(
        AIModelClientFactory,
        "create_provider_client",
        classmethod(make_fake_provider()),
    )

    client = AIModelClient(
        model_endpoint=text_endpoint,
        config=config,
        is_async=False,
    )

    response = client.generate(prompt="trigger logging")

    assert response.status is not None
    assert response.status.status == ExternalApiCallStatusEnum.RESPONSE_RECEIVED_SUCCESS

    log_file = Path(tmp_path) / "case_dai_018" / "dai" / "dai_python_logs.jsonl"
    assert log_file.exists()

    contents = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert contents  # Expect at least the start/stop bookkeeping entries

    # Each line should be valid JSON payload
    for line in contents:
        json.loads(line)
