"""Artifact configuration for capturing AI model call details."""

from pathlib import Path

from pydantic import Field

from dhenara.ai.types.shared.base import BaseModel


class ArtifactConfig(BaseModel):
    """Configuration for capturing AI model call artifacts.

    Controls what gets logged during AI model calls for debugging and analysis.
    When enabled, captures request/response data at both dhenara-ai and provider levels.

    Attributes:
        enabled: Master switch for artifact capture
        artifact_root: Base directory path for storing artifacts (required if enabled)
        capture_dhenara_request: Capture high-level dhenara-ai request (prompt, messages, config)
        capture_provider_request: Capture provider-specific formatted request payloads
        capture_provider_response: Capture raw provider API response before parsing
        capture_dhenara_response: Capture parsed dhenara-ai response objects
        prefix: Optional prefix for artifact filenames (e.g., "step_001_execute")
    """

    enabled: bool = Field(
        default=False,
        description="Enable artifact capture",
    )
    artifact_root: str | Path | None = Field(
        default=None,
        description="Root directory path for storing artifacts (required if enabled)",
    )
    capture_dhenara_request: bool = Field(
        default=True,
        description="Capture dhenara-ai request (prompt/messages/config)",
    )
    capture_provider_request: bool = Field(
        default=True,
        description="Capture provider-specific formatted request",
    )
    capture_provider_response: bool = Field(
        default=True,
        description="Capture raw provider response",
    )
    capture_dhenara_response: bool = Field(
        default=True,
        description="Capture parsed dhenara-ai response",
    )
    prefix: str | None = Field(
        default=None,
        description="Optional prefix for artifact filenames",
    )
    enable_python_logs: bool = Field(
        default=False,
        description="Capture Python logging records for this call",
    )
    python_log_level: str | int = Field(
        default="INFO",
        description="Logging level for captured Python logs (e.g., DEBUG, INFO, WARNING)",
    )
    python_logger_levels: dict[str, str | int] | None = Field(
        default=None,
        description="Optional overrides for specific loggers (name -> level or 'OFF')",
    )

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        if self.enabled and not self.artifact_root:
            raise ValueError("artifact_root is required when artifacts are enabled")
        super().model_post_init(__context)
