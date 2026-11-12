"""CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R008: destructive operations use the SDK's one declaration and guard."""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import patch

import pytest
import requests
from fastmcp import Client, FastMCP

pytestmark = pytest.mark.spec("CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R008")

DECLARED = {
    "github_merge_pull_request",
    "github_update_dependabot_alert",
    "github_pulls_enable_auto_merge",
}
CLASSIFIED = {"github_delete_repository", "github_delete_branch"}


def _server() -> FastMCP:
    module = importlib.import_module("github_agent.mcp_server")
    with patch(
        "sys.argv", ["github-mcp", "--transport", "stdio", "--auth-type", "none"]
    ):
        result = module.get_mcp_instance()
    if isinstance(result, FastMCP):
        return result
    return next(value for value in result if isinstance(value, FastMCP))


def test_declared_and_classified_operations_are_destructive_tools_that_ask_first(
    monkeypatch,
):
    monkeypatch.delenv("SCION_ALLOW_DESTRUCTIVE", raising=False)

    async def _run():
        async with Client(_server()) as client:
            tools = {t.name: t for t in await client.list_tools()}
            for name in DECLARED | CLASSIFIED:
                assert tools[name].annotations.destructive_hint is True, name
                assert "confirm" in tools[name].input_schema["properties"], name
            for name, tool in tools.items():
                assert "allow_destructive" not in tool.input_schema["properties"], name

    asyncio.run(_run())


@pytest.mark.parametrize(
    "name,arguments",
    [
        (
            "github_merge_pull_request",
            {"owner": "o", "repo": "r", "number": 1},
        ),
        (
            "github_delete_branch",
            {"owner": "o", "repo": "r", "branch": "b"},
        ),
    ],
)
def test_unconfirmed_call_sends_no_request(name, arguments, monkeypatch):
    monkeypatch.delenv("SCION_ALLOW_DESTRUCTIVE", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    sent: list[str] = []

    def record(self, method, url, **kwargs):
        sent.append(f"{method} {url}")
        raise AssertionError("no request may be made without confirmation")

    monkeypatch.setattr(requests.Session, "request", record)

    async def _run():
        async with Client(_server()) as client:
            refused = await client.call_tool(name, arguments, raise_on_error=False)
            assert refused.structured_content["cancelled"] is True

    asyncio.run(_run())
    assert sent == []


def test_no_connector_local_knob_remains():
    auth = importlib.import_module("github_agent.auth")
    assert not hasattr(auth, "allow_destructive_default")
