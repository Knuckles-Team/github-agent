"""CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R001: the server lists its packaged skills, ontology and prompts by default."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest
from fastmcp import Client

PACKAGE = "github_agent"
ROOT = Path(list(importlib.import_module(PACKAGE).__path__)[0])

pytestmark = pytest.mark.spec("CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R001")


def _server():
    from unittest.mock import patch

    from fastmcp import FastMCP

    module = importlib.import_module("github_agent.mcp_server")
    with patch(
        "sys.argv", ["github-mcp", "--transport", "stdio", "--auth-type", "none"]
    ):
        result = module.get_mcp_instance()
    if isinstance(result, FastMCP):
        return result
    return next(value for value in result if isinstance(value, FastMCP))


async def _listings():
    async with Client(_server()) as client:
        resources = {str(r.uri) for r in await client.list_resources()}
        prompts = {p.name for p in await client.list_prompts()}
        for uri in resources:
            if uri.startswith(("ontology://", "shapes://")):
                directory = ROOT / "ontology"
                if uri.startswith("shapes://"):
                    directory = directory / "shapes"
                actual = await client.read_resource(uri)
                assert (
                    actual[0].text.encode()
                    == (directory / uri.rsplit("/", 1)[-1]).read_bytes()
                )
    return resources, prompts


def test_default_content_surfaces() -> None:
    resources, prompts = asyncio.run(_listings())
    if list(ROOT.glob("skills/*/SKILL.md")):
        assert any(u.startswith("skill://") for u in resources)
    assert list(ROOT.glob("ontology/*.ttl"))
    assert any(u.startswith("ontology://github-mcp/") for u in resources)
    for path in ROOT.glob("prompts/*.json"):
        if not path.name.startswith("_"):
            assert path.stem in prompts


@pytest.fixture(autouse=True)
def _no_outbound_network(monkeypatch):
    import socket

    def refuse(*_args, **_kwargs):
        raise AssertionError("content publication must not contact an external system")

    monkeypatch.setenv("MCP_FLEET_REGISTRATION", "false")
    monkeypatch.setattr(socket.socket, "connect", refuse)
