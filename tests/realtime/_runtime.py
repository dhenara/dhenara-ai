from __future__ import annotations

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest


@dataclass
class ScenarioRecord:
    scenario: str
    provider: str
    model: str
    status: str
    message: str
    duration: float


@dataclass
class ProviderSummary:
    counts: dict[str, int] = field(default_factory=lambda: {"pass": 0, "fail": 0, "skip": 0})
    models: dict[str, dict[str, int]] = field(default_factory=dict)


_SCENARIO_RECORDS: list[ScenarioRecord] = []
_PROVIDER_SUMMARY: dict[str, ProviderSummary] = defaultdict(ProviderSummary)


def _increment(provider: str, model: str, status: str) -> None:
    summary = _provider_summary_entry(provider)
    summary.counts[status] = summary.counts.get(status, 0) + 1
    model_counts = summary.models.setdefault(model, {"pass": 0, "fail": 0, "skip": 0})
    model_counts[status] = model_counts.get(status, 0) + 1


def _provider_summary_entry(provider: str) -> ProviderSummary:
    if provider not in _PROVIDER_SUMMARY:
        _PROVIDER_SUMMARY[provider] = ProviderSummary()
    return _PROVIDER_SUMMARY[provider]


def _record(record: ScenarioRecord) -> None:
    _SCENARIO_RECORDS.append(record)
    _increment(record.provider, record.model, record.status)


def _resolver(endpoint, provider: str | None, model: str | None) -> tuple[str, str]:
    if endpoint is not None:
        provider = getattr(endpoint.api.provider, "value", str(endpoint.api.provider))
        ai_model = getattr(endpoint.ai_model, "model_name", None)
        model = getattr(endpoint.ai_model, "model_name_with_version_suffix", ai_model or "unknown")
    return provider or "unknown", model or "unknown"


def _log(prefix: str, scenario: str, provider: str, model: str, message: str = "") -> None:
    text = f"[realtime] {prefix} {scenario} @ {provider}/{model}"
    if message:
        text = f"{text} :: {message}"
    print(text, flush=True, file=sys.stderr)


@contextmanager
def track_scenario(endpoint, scenario: str, *, provider: str | None = None, model: str | None = None):
    provider_name, model_name = _resolver(endpoint, provider, model)
    start = time.perf_counter()
    _log("▶", scenario, provider_name, model_name)
    try:
        yield
    except pytest.skip.Exception as exc:  # pragma: no cover - depends on runtime env
        duration = time.perf_counter() - start
        _record(
            ScenarioRecord(
                scenario=scenario,
                provider=provider_name,
                model=model_name,
                status="skip",
                message=str(exc),
                duration=duration,
            )
        )
        _log("⚠", scenario, provider_name, model_name, str(exc))
        raise
    except AssertionError as exc:
        duration = time.perf_counter() - start
        _record(
            ScenarioRecord(
                scenario=scenario,
                provider=provider_name,
                model=model_name,
                status="fail",
                message=str(exc),
                duration=duration,
            )
        )
        _log("✖", scenario, provider_name, model_name, str(exc))
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        duration = time.perf_counter() - start
        _record(
            ScenarioRecord(
                scenario=scenario,
                provider=provider_name,
                model=model_name,
                status="fail",
                message=repr(exc),
                duration=duration,
            )
        )
        _log("✖", scenario, provider_name, model_name, repr(exc))
        raise
    else:
        duration = time.perf_counter() - start
        _record(
            ScenarioRecord(
                scenario=scenario,
                provider=provider_name,
                model=model_name,
                status="pass",
                message="",
                duration=duration,
            )
        )
        _log("✓", scenario, provider_name, model_name)


def build_provider_summary() -> str:
    if not _SCENARIO_RECORDS:
        return ""
    lines = ["", "Realtime provider summary:"]
    for provider in sorted(_PROVIDER_SUMMARY.keys()):
        summary = _PROVIDER_SUMMARY[provider]
        counts = summary.counts
        lines.append(
            f"- {provider}: pass={counts.get('pass', 0)} fail={counts.get('fail', 0)} skip={counts.get('skip', 0)}"
        )
        for model in sorted(summary.models.keys()):
            model_counts = summary.models[model]
            lines.append(
                f"    • {model}: pass={model_counts.get('pass', 0)} fail={model_counts.get('fail', 0)} "
                f"skip={model_counts.get('skip', 0)}"
            )
    return "\n".join(lines)


def get_scenario_records() -> list[ScenarioRecord]:
    return list(_SCENARIO_RECORDS)
