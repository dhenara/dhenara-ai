from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path


class ArtifactKind(str, Enum):
    LOG = "log"
    PAYLOAD = "payload"
    SNAPSHOT = "snapshot"
    MEDIA = "media"
    UNKNOWN = "unknown"


@dataclass
class ArtifactMetadata:
    path: Path
    kind: ArtifactKind
    label: str


_RUN_ID_ENV = "DAI_TEST_ARTIFACT_RUN_ID"


def _generate_run_id() -> str:
    run_id = os.environ.get(_RUN_ID_ENV)
    if run_id:
        return run_id
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    os.environ[_RUN_ID_ENV] = timestamp
    return timestamp


def artifact_run_directory(base_root: Path | None = None) -> Path:
    base_dir = (
        Path(base_root) if base_root else Path(os.environ.get("DAI_TEST_ARTIFACT_DIR", Path("/tmp/dvi_artifacts")))
    )
    run_id = _generate_run_id()
    run_dir = base_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class TestArtifactManager:
    """Normalize artifact handling for realtime-style tests."""

    def __init__(
        self,
        base_dir: Path,
        max_files: int = 100,
        per_run_limit: int = 10,
        retain_kinds: Iterable[ArtifactKind] | None = None,
        cleanup_enabled: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.max_files = max_files
        self.per_run_limit = per_run_limit
        self.retain_kinds = set(retain_kinds or [])
        self.cleanup_enabled = cleanup_enabled
        self._artifacts: dict[str, list[ArtifactMetadata]] = {}
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def register(self, run_id: str, metadata: ArtifactMetadata) -> Path:
        run_bucket = self._artifacts.setdefault(run_id, [])
        if len(run_bucket) >= self.per_run_limit:
            return metadata.path
        run_bucket.append(metadata)
        return metadata.path

    def flush(self) -> None:
        if not self.cleanup_enabled or self.max_files <= 0:
            return
        tracked: list[ArtifactMetadata] = [item for items in self._artifacts.values() for item in items]
        tracked.sort(key=lambda meta: meta.path.stat().st_mtime if meta.path.exists() else 0)
        while len(tracked) > self.max_files:
            meta = tracked.pop(0)
            if self.retain_kinds and meta.kind in self.retain_kinds:
                continue
            try:
                meta.path.unlink(missing_ok=True)
            except OSError:
                pass

    def list_artifacts(self, run_id: str | None = None) -> dict[str, list[ArtifactMetadata]]:
        if run_id is None:
            return self._artifacts
        return {run_id: self._artifacts.get(run_id, [])}


def default_artifact_manager() -> TestArtifactManager:
    run_dir = artifact_run_directory()
    max_files = int(os.environ.get("DAI_TEST_ARTIFACT_MAX_FILES", "200"))
    per_run_limit = int(os.environ.get("DAI_TEST_ARTIFACT_PER_RUN", "20"))
    retain_env = os.environ.get("DAI_TEST_ARTIFACT_RETAIN", "")
    retain_kinds = [ArtifactKind(item) for item in retain_env.split(",") if item]
    cleanup_flag = os.environ.get("DAI_TEST_ARTIFACT_CLEANUP", "1").lower()
    cleanup_enabled = cleanup_flag not in {"0", "false", "no"}
    return TestArtifactManager(
        base_dir=run_dir,
        max_files=max_files,
        per_run_limit=per_run_limit,
        retain_kinds=retain_kinds,
        cleanup_enabled=cleanup_enabled,
    )
