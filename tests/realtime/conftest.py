from ._artifacts import get_artifact_manager
from ._config import realtime_resource_config  # noqa: F401
from ._runtime import build_provider_summary


def pytest_sessionfinish(session, exitstatus):
    summary = build_provider_summary()
    if summary:
        print(summary)
    get_artifact_manager().flush()
