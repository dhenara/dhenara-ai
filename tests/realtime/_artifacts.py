from __future__ import annotations

from dhenara.ai.testing import TestArtifactManager, default_artifact_manager

ARTIFACT_MANAGER: TestArtifactManager = default_artifact_manager()


def get_artifact_manager() -> TestArtifactManager:
    return ARTIFACT_MANAGER
