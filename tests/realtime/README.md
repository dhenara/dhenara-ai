# Realtime Test Guide

## Provider/Model Coverage

Realtime pytest coverage is split across two suites:

- `test_all_capabilities.py` iterates over the provider/model pairs in `DEFAULT_PROVIDER_MODELS_MAP`.
- `test_all_providers.py` runs a structured-output sweep across every distinct configured provider.

The provider/model maps live in `_config.py`. To onboard a new model, add it to the appropriate map and ensure `ResourceConfig` can expose a matching endpoint.

Realtime credentials follow the same package contract as normal runtime code:

- pass an explicit credentials file path into `ResourceConfig.load_from_file(...)`, or
- set `DAI_SECRET_CONFIG_DIR` and place `dai_credentials.yaml` in that directory.

If `DAI_SECRET_CONFIG_DIR` is unset, the default lookup path is `/run/secrets/dai/dai_credentials.yaml`.

## Artifact Storage

Test artifacts (request/response captures, payloads, etc.) are managed by the
shared `TestArtifactManager` under `dhenara.ai.testing`. Key environment knobs:

- `DAI_TEST_ARTIFACT_DIR` (default: `/tmp/dvi_artifacts`): root directory for all
  recorded artifacts. Inspect this path after `pytest` to review logs or payloads
  captured during the latest run.
- Each pytest invocation creates a unique `run_<timestamp>` folder within
  `DAI_TEST_ARTIFACT_DIR`. When tests execute inside an individual package
  (e.g., `packages/dhenara_ai` or `verif_angels/verifinder`), artifacts land
  under `run_<timestamp>/<suite>/...`. Override `DAI_TEST_ARTIFACT_SUITE` to
  force a custom suite directory name when needed.
- `DAI_TEST_ARTIFACT_MAX_FILES` (default: `200`): upper bound for tracked files
  before cleanup begins.
- `DAI_TEST_ARTIFACT_PER_RUN` (default: `20`): max artifacts tracked per scenario
  run ID.
- `DAI_TEST_ARTIFACT_CLEANUP` (default: `1`): set to `0`, `false`, or `no` to
  disable automatic cleanup and retain every artifact in `DAI_TEST_ARTIFACT_DIR`.

When artifacts are enabled, the same run directory also receives `pytest.log`
and `status.txt`. The log aggregates captured stdout/stderr for every test, and
the status file summarizes pass/fail/skipped nodes along with failure messages
to speed up triage without rerunning the suite.

To inspect artifacts after a run, list the directory referenced by
`DAI_TEST_ARTIFACT_DIR` (or the default path) and drill into the scenario and
label subdirectories created by `ArtifactTracker`.
