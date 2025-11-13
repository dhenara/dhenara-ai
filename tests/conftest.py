import pytest


@pytest.fixture(scope="session")
def package_name() -> str:
    """Marker fixture to identify the package under test."""
    return "dhenara-ai"
