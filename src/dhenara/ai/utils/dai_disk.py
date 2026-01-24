"""Small, dependency-free disk + JSON helpers for Dhenara AI (DAI).

Why this exists
- Keeps callsites consistent (one place to harden defaults like encoding).
- Avoids sprinkling direct stdlib `json.*` / `open()` usage across the codebase.
- Stays tiny and provider-agnostic so it can live in the public `dhenara_ai` package.

Design goals
- Minimal API surface (only what we actually need).
- Accept `str | Path` in most places.
- Keep JSON and disk I/O separate but composable.

Notes
- This module is intentionally synchronous.
- JSONL helpers treat `str` rows as already-serialized lines (written as-is).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

type Pathish = str | Path


class DaiJsonOps:
    """Wrapper around stdlib `json`.

    Call sites should prefer `DAI_JSON` instead of using `json.*` directly.
    """

    JSONDecodeError = json.JSONDecodeError

    def loads(self, s: str, **kwargs: Any) -> Any:
        return json.loads(s, **kwargs)

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        return json.dumps(obj, **kwargs)

    def load(self, fp: IO[str], **kwargs: Any) -> Any:
        return json.load(fp, **kwargs)

    def dump(self, obj: Any, fp: IO[str], **kwargs: Any) -> Any:
        return json.dump(obj, fp, **kwargs)


class DaiDiskOps:
    """Minimal disk helpers.

    Prefer these over direct `Path.read_text` / `Path.write_text` / `open()`.
    """

    def __init__(
        self,
        *,
        default_encoding: str = "utf-8",
        default_errors: str = "strict",
        json_ops: DaiJsonOps | None = None,
    ) -> None:
        self._default_encoding = default_encoding
        self._default_errors = default_errors
        self._json = json_ops or DaiJsonOps()

    def as_path(self, p: Pathish) -> Path:
        return p if isinstance(p, Path) else Path(p)

    def ensure_dir(self, p: Pathish) -> Path:
        path = self.as_path(p)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_parent(self, p: Pathish) -> Path:
        path = self.as_path(p)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return path

    def exists(self, p: Pathish) -> bool:
        try:
            return self.as_path(p).exists()
        except Exception:
            return False

    def is_file(self, p: Pathish) -> bool:
        try:
            return self.as_path(p).is_file()
        except Exception:
            return False

    def is_dir(self, p: Pathish) -> bool:
        try:
            return self.as_path(p).is_dir()
        except Exception:
            return False

    def read_text(self, p: Pathish, *, encoding: str | None = None, errors: str | None = None) -> str:
        path = self.as_path(p)
        return path.read_text(encoding=encoding or self._default_encoding, errors=errors or self._default_errors)

    def read_bytes(self, p: Pathish) -> bytes:
        return self.as_path(p).read_bytes()

    def write_text(
        self,
        p: Pathish,
        text: str,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        ensure_parent: bool = True,
    ) -> Path:
        path = self.as_path(p)
        if ensure_parent:
            self.ensure_parent(path)
        path.write_text(text, encoding=encoding or self._default_encoding, errors=errors or self._default_errors)
        return path

    def write_bytes(self, p: Pathish, data: bytes, *, ensure_parent: bool = True) -> Path:
        path = self.as_path(p)
        if ensure_parent:
            self.ensure_parent(path)
        path.write_bytes(data)
        return path

    def open_text(
        self,
        p: Pathish,
        mode: str = "r",
        *,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> IO[str]:
        path = self.as_path(p)
        if any(flag in mode for flag in ("w", "a", "+")):
            self.ensure_parent(path)
        return path.open(mode, encoding=encoding or self._default_encoding, errors=errors or self._default_errors)

    def open_bytes(self, p: Pathish, mode: str = "rb") -> IO[bytes]:
        path = self.as_path(p)
        if any(flag in mode for flag in ("w", "a", "+")):
            self.ensure_parent(path)
        return path.open(mode)

    def unlink(self, p: Pathish, *, missing_ok: bool = True) -> None:
        path = self.as_path(p)
        try:
            path.unlink(missing_ok=missing_ok)
        except TypeError:
            if missing_ok:
                try:
                    path.unlink()
                except FileNotFoundError:
                    return
            else:
                path.unlink()

    def stat_mtime(self, p: Pathish) -> float:
        try:
            return float(self.as_path(p).stat().st_mtime)
        except Exception:
            return 0.0

    def glob(self, dir_path: Pathish, pattern: str) -> list[Path]:
        """Best-effort glob within a directory."""

        try:
            return list(self.as_path(dir_path).glob(pattern))
        except Exception:
            return []

    def touch(self, p: Pathish) -> Path:
        path = self.as_path(p)
        self.ensure_parent(path)
        try:
            path.touch(exist_ok=True)
        except Exception:
            pass
        return path

    def read_json(
        self, p: Pathish, *, encoding: str | None = None, errors: str | None = None, **json_kwargs: Any
    ) -> Any:
        with self.open_text(p, "r", encoding=encoding, errors=errors) as fp:
            return self._json.load(fp, **json_kwargs)

    def write_json(
        self,
        p: Pathish,
        obj: Any,
        *,
        encoding: str | None = None,
        ensure_parent: bool = True,
        **json_kwargs: Any,
    ) -> Path:
        path = self.as_path(p)
        if ensure_parent:
            self.ensure_parent(path)
        with path.open("w", encoding=encoding or self._default_encoding) as fp:
            self._json.dump(obj, fp, **json_kwargs)
        return path

    def write_jsonl(
        self,
        p: Pathish,
        rows: Iterable[Any],
        *,
        encoding: str | None = None,
        append: bool = False,
        ensure_parent: bool = True,
        **json_kwargs: Any,
    ) -> Path:
        path = self.as_path(p)
        if ensure_parent:
            self.ensure_parent(path)
        mode = "a" if append else "w"
        with path.open(mode, encoding=encoding or self._default_encoding) as fp:
            for row in rows:
                if isinstance(row, str):
                    fp.write(row)
                else:
                    fp.write(self._json.dumps(row, **json_kwargs))
                fp.write("\n")
        return path

    def iter_lines(self, p: Pathish, *, encoding: str | None = None, errors: str | None = None) -> Iterator[str]:
        with self.open_text(p, "r", encoding=encoding, errors=errors) as fp:
            yield from fp

    def iter_jsonl(
        self,
        p: Pathish,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        **json_kwargs: Any,
    ) -> Iterator[Any]:
        for line in self.iter_lines(p, encoding=encoding, errors=errors):
            s = line.strip()
            if not s:
                continue
            yield self._json.loads(s, **json_kwargs)


DAI_JSON = DaiJsonOps()
DAI_DISK = DaiDiskOps(json_ops=DAI_JSON)
