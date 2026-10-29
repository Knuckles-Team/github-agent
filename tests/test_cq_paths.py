"""Competency-question paths of the GitHub pack (CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R004)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from agent_connector_sdk.ingest.live_declaration import load_declaration

import github_agent

pytestmark = pytest.mark.spec("CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R004")

PACKAGE = Path(github_agent.__file__).parent
TTL = (PACKAGE / "ontology" / "github.ttl").read_text("utf-8")
# Concepts the epistemic-graph ontology owns: referenced, never declared here.
SHARED = ("Repository", "PullRequest", "Release", "Branch", "Commit", "PipelineRun")


def _block(name: str) -> str:
    match = re.search(rf"^:{name} a [^\n]*\n(?:[^\n]+\n)*", TTL, flags=re.M)
    assert match, name
    return match.group(0)


@pytest.mark.parametrize("name", SHARED)
def test_shared_concepts_are_not_redeclared(name: str) -> None:
    assert not re.search(rf"^:{name} a owl:Class", TTL, flags=re.M), name


def test_state_is_declared_on_the_issue_type() -> None:
    block = _block("state")
    assert re.search(r"rdfs:domain \[ a owl:Class ; owl:unionOf \( :Issue ", block)


def test_check_run_extends_pipeline_run() -> None:
    assert "rdfs:subClassOf :PipelineRun" in _block("CheckRun")


def test_no_mapping_has_a_null_ontology_class() -> None:
    unmapped = [
        name
        for name, mapping in load_declaration("github_agent").schema_mappings.items()
        if not mapping.ontology_class
    ]
    assert unmapped == []
