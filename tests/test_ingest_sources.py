"""The packaged ingestible-source declaration (CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003)."""

from __future__ import annotations

import pytest
from agent_connector_sdk.ingest.live_declaration import load_declaration
from agent_connector_sdk.mcp.method_tools import operation_annotations, operation_verb
from agent_connector_sdk.mcp.tool_surface import operation_prefix

from github_agent.api_client import Api


@pytest.mark.spec("CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003")
def test_declares_ingestible_sources() -> None:
    declaration = load_declaration("github_agent")
    assert declaration is not None
    assert declaration.connector == "github-mcp"
    source = declaration.source("issues")
    assert source is not None and source.record_type == "Issue"
    prefix = operation_prefix("github-agent")
    assert source.tools == (f"{prefix}_search_issues",)
    # The engine admits a record only under the pack mapping of its type.
    assert source.record_type in load_declaration("github_agent").schema_mappings
    method = Api.search_issues
    annotations = operation_annotations("search_issues", operation_verb(method))
    assert annotations.read_only_hint is True
