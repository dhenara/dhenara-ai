"""Compatibility shim.

Prefer importing from `dhenara.ai.utils.dai_disk`:
- `DAI_DISK`, `DAI_JSON`

This module remains to avoid breaking older imports.
"""

from .dai_disk import (
    DAI_DISK,
    DAI_JSON,
    DaiDiskOps,
    DaiJsonOps,
)

# Backwards-compatible names
AiDiskOps = DaiDiskOps
AiJsonOps = DaiJsonOps
AI_JSON = DAI_JSON
AI_DISK = DAI_DISK
