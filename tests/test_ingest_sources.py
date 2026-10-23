"""The packaged ingestible-source declaration (CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from agent_connector_sdk.ingest.sources import load_ingest_sources
from agent_connector_sdk.mcp.method_tools import operation_annotations, operation_verb
from agent_connector_sdk.mcp.tool_surface import operation_prefix

import github_agent
from github_agent.api_client import Api


@pytest.mark.spec("CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003")
def test_declares_ingestible_sources() -> None:
    package_root = Path(github_agent.__file__).parent
    declaration = load_ingest_sources(package_root)
    assert declaration is not None
    assert declaration.connector == "github-mcp"
    source = declaration.source("issues")
    assert source is not None and source.record_type == "Issue"
    prefix = operation_prefix("github-agent")
    assert source.tools == (f"{prefix}_search_issues",)
    # The engine admits a record only under the pack mapping of its type.
    manifest = yaml.safe_load(
        (package_root.parent / "connector_manifest.yml").read_text("utf-8")
    )
    assert source.record_type in manifest["schema_mappings"]
    method = getattr(Api, "search_issues")
    annotations = operation_annotations("search_issues", operation_verb(method))
    assert annotations.read_only_hint is True
