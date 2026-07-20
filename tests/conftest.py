"""Shared test fixtures for Github Agent."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("GITHUB_URL", "https://test.example.com")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-12345")
