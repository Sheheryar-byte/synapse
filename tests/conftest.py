"""Shared pytest configuration."""
import pytest


# pytest-asyncio: auto mode set in pyproject.toml
# No additional fixtures needed for v0.1

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests (deselect with -m 'not integration')")
