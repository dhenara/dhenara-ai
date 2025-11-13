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

            def _json_default(obj):
                """Custom JSON serializer that handles Pydantic models and SDK objects."""
                # Try model_dump for Pydantic v2 models and SDK objects
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                # Try dict() for Pydantic v1 models
                if hasattr(obj, "dict") and callable(obj.dict):
                    return obj.dict()
                # Fallback to string representation
                return str(obj)

            # Write UTF-8 JSON, preserve unicode (ensure_ascii=False) for readability
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    default=_json_default,
                    ensure_ascii=False,  # keep characters like smart quotes unescaped
                )

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

    @staticmethod
    def write_jsonl(
        artifact_root: Path,
        filename: str,
        rows: list[dict] | list[str],
        prefix: str | None = None,
    ) -> None:
        """Write newline-delimited JSON (JSONL) artifact to disk.

        Args:
            artifact_root: Root directory for artifacts
            filename: Name of the artifact file (e.g., 'records.jsonl')
            rows: List of dicts or JSON-serializable items to write as one JSON per line
            prefix: Optional prefix for subdirectory
        """
        try:
            artifact_root = Path(artifact_root)

            artifact_dir = artifact_root / prefix if prefix else artifact_root
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / filename

            # Pre-serialize rows to avoid exceptions in the tight write loop
            # Preserve unicode in JSONL lines as well
            serialized: list[str] = [
                json.dumps(
                    row,
                    default=str,
                    ensure_ascii=False,
                )
                for row in (rows or [])
            ]

            with open(artifact_path, "w", encoding="utf-8") as f:
                for line in serialized:
                    f.write(line)
                    f.write("\n")

            logger.debug(f"Wrote artifact: {artifact_path}")

        except Exception as e:
            logger.warning(f"Failed to write artifact {filename}: {e}")

    @staticmethod
    def append_jsonl(
        artifact_root: Path,
        filename: str,
        rows: list[dict] | list[str],
        prefix: str | None = None,
    ) -> None:
        """Append rows to a newline-delimited JSON (JSONL) artifact.

        Args:
            artifact_root: Root directory for artifacts
            filename: Name of the artifact file (e.g., 'records.jsonl')
            rows: List of dicts or JSON-serializable items to append
            prefix: Optional prefix for subdirectory
        """
        try:
            artifact_root = Path(artifact_root)

            artifact_dir = artifact_root / prefix if prefix else artifact_root
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / filename

            serialized: list[str] = [json.dumps(row, default=str, ensure_ascii=False) for row in (rows or [])]

            with open(artifact_path, "a", encoding="utf-8") as f:
                for line in serialized:
                    f.write(line)
                    f.write("\n")

            logger.debug(f"Appended artifact rows: {artifact_path}")

        except Exception as e:
            logger.warning(f"Failed to append artifact {filename}: {e}")
