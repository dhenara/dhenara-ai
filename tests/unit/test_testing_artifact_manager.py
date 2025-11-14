from __future__ import annotations

from dhenara.ai.testing import ArtifactKind, ArtifactMetadata
from dhenara.ai.testing import TestArtifactManager as ArtifactManager


def test_register_and_flush(tmp_path):
    manager = ArtifactManager(base_dir=tmp_path, max_files=1, per_run_limit=2)

    first = tmp_path / "run_a" / "first.log"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("a")
    manager.register("run_a", ArtifactMetadata(path=first, kind=ArtifactKind.LOG, label="first"))

    second = tmp_path / "run_b" / "second.log"
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_text("b")
    manager.register("run_b", ArtifactMetadata(path=second, kind=ArtifactKind.LOG, label="second"))

    manager.flush()

    assert not first.exists()
    assert second.exists()


def test_per_run_limit(tmp_path):
    manager = ArtifactManager(base_dir=tmp_path, max_files=10, per_run_limit=1)
    path_one = tmp_path / "run" / "one.log"
    path_one.parent.mkdir(parents=True, exist_ok=True)
    path_one.touch()
    manager.register("same_run", ArtifactMetadata(path=path_one, kind=ArtifactKind.LOG, label="one"))

    path_two = tmp_path / "run" / "two.log"
    path_two.touch()
    manager.register("same_run", ArtifactMetadata(path=path_two, kind=ArtifactKind.LOG, label="two"))

    tracked = manager.list_artifacts("same_run")["same_run"]
    assert len(tracked) == 1
    assert tracked[0].path == path_one
