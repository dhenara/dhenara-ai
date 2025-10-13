"""Utilities for capturing and persisting artifacts during AI model calls."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ArtifactWriter:
    """Handles writing of artifacts during AI model execution."""

    @staticmethod
    def write_json(artifact_root: Path, filename: str, data: Any, prefix: str | None = None) -> None:
        """Write JSON artifact to disk.

        Args:
            artifact_root: Root directory for artifacts
            filename: Name of the artifact file (e.g., 'dhenara_request.json')
            data: Data to serialize (must be JSON-serializable)
            prefix: Optional prefix for subdirectory (e.g., 'step_001_execute_loop_002')
        """
        try:
            artifact_root = Path(artifact_root)

            # Create subdirectory if prefix provided
            if prefix:
                artifact_dir = artifact_root / prefix
            else:
                artifact_dir = artifact_root

            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / filename

            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Wrote artifact: {artifact_path}")

        except Exception as e:
            logger.warning(f"Failed to write artifact {filename}: {e}")

    @staticmethod
    def write_text(artifact_root: Path, filename: str, content: str, prefix: str | None = None) -> None:
        """Write text artifact to disk.

        Args:
            artifact_root: Root directory for artifacts
            filename: Name of the artifact file (e.g., 'provider_request.txt')
            content: Text content to write
            prefix: Optional prefix for subdirectory
        """
        try:
            artifact_root = Path(artifact_root)

            if prefix:
                artifact_dir = artifact_root / prefix
            else:
                artifact_dir = artifact_root

            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / filename

            with open(artifact_path, "w") as f:
                f.write(content)

            logger.debug(f"Wrote artifact: {artifact_path}")

        except Exception as e:
            logger.warning(f"Failed to write artifact {filename}: {e}")
