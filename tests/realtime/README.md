# Realtime Test Guide

## Provider/Model Coverage

Realtime pytest cases now iterate over every provider/model pair defined in
`PROVIDER_MODEL_PREFERENCE` (see `_config.py`). For each configured provider the
active model names (uncommented entries) are located, and every scenario in
`test_examples.py` runs against the resulting endpoint. To onboard a new model,
add it to the preference map and ensure `ResourceConfig` exposes a matching
endpoint.

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
