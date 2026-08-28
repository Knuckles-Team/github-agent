"""Native epistemic-graph typed-node + document ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` / ``ingest_documents`` seam and the GitHub
record mappers with a fake ChangeEnvelope-capable engine client (no engine required),
asserting the committed nodes/edges and the repository/PR/issue/release mappings.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.

The fake client mirrors agent-utilities' own sanctioned test double
(``agent-utilities/tests/knowledge_graph/test_native_ingest.py``) — the
``txn``-only fake is retired; ``native_ingest`` now hard-requires an injected
client exposing ``.changes``/``.nodes``/``.rdf``/``.supports()``.
"""

from __future__ import annotations

from typing import Any

import msgpack
import pytest

# `agent_utilities.knowledge_graph.memory` unconditionally imports
# `agent_utilities.numeric` at module-load time, which in turn requires the
# compiled `epistemic_graph.numeric` kernel. agent-utilities moved that
# kernel out of its base dependency set into the opt-in `graphos` extra
# (GOC-73); this repo depends only on `agent-utilities[mcp]`, which does not
# pull it in. `github_agent.kg_ingest` (imported below) transitively imports
# `agent_utilities.knowledge_graph.memory.native_ingest`, so left unguarded
# this raises a bare ModuleNotFoundError/ImportError chain that pytest
# reports as a COLLECTION ERROR — which (a) reads exactly like a regression
# in THIS repo and (b) aborts collection of the entire `tests/` suite, not
# just this file (`pytest tests/ -q` reports "0 tests collected, 1 error"
# for the whole run, which is why lanes have been passing
# `--ignore=tests/test_kg_ingest.py` and silently losing coverage on both
# sides of every before/after comparison). This is an ENVIRONMENT/packaging
# gap, not application-code breakage — install
# `agent-utilities[graphos]>=2.27.0` to exercise these tests. See
# plans/complex/waves/wD4/WD4-FIX-01.md defect (d). Turn it into a clean,
# LOUD, explained skip of just this file instead.
pytest.importorskip(
    "agent_utilities.knowledge_graph.memory.native_ingest",
    # pytest 9.1 changed importorskip()'s default `exc_type` from
    # ImportError to ModuleNotFoundError (see the versionchanged note in
    # pytest.importorskip's own docstring). agent_utilities.numeric
    # deliberately re-raises a plain ImportError (not ModuleNotFoundError)
    # with an explanatory message, so the new default silently fails to
    # catch it and the "skip" degrades right back into the collection
    # error this guard exists to prevent. Pin exc_type explicitly so the
    # guard keeps working regardless of installed pytest version.
    exc_type=ImportError,
    reason=(
        "agent_utilities.numeric requires the compiled epistemic_graph.numeric "
        "kernel, shipped only behind agent-utilities' opt-in `graphos` extra "
        "(GOC-73); not installed by this repo's `agent-utilities[mcp]` "
        "dependency — install `agent-utilities[graphos]>=2.27.0` to run "
        "KG-ingestion tests (WD4-FIX-01 defect (d))"
    ),
)

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

from github_agent.kg_ingest import (
    ingest_entities,
    ingest_issues,
    ingest_pipeline_runs,
    ingest_pull_requests,
    ingest_release_notes,
    ingest_repositories,
)


@pytest.fixture(autouse=True)
def _governed_session():
    actor = ActorContext(
        actor_id="subject:opaque:synthetic",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=(),
        tenant_id="tenant:opaque:synthetic",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:write"}),
        graph="graph:opaque:synthetic",
        policy_version="policy:opaque:synthetic",
        audience="epistemic-graph",
    )
    with use_actor(actor), use_session(session):
        yield


class _FakeNodes:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}

    def properties(self, node_id: str) -> dict[str, Any] | None:
        return self.values.get(node_id)

    def list(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self.values.items())


class _FakeChanges:
    def __init__(self, nodes: _FakeNodes) -> None:
        self.nodes = nodes
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self.applied: list[dict[str, Any]] = []
        self.records: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, dict[str, Any]] = {}

    def get(self, envelope_id: str) -> dict[str, Any] | None:
        return self.records.get(envelope_id)

    def content_version(self, object_id: str) -> dict[str, Any] | None:
        return self.versions.get(object_id)

    def cursor(self, _source: str, _partition: str = "") -> None:
        return None

    def apply(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self.applied.append(envelope)
        mutation = envelope["mutation"]
        for operation in mutation["operations"]:
            method = operation["method"]
            params = method["params"]
            properties = msgpack.unpackb(params["properties_msgpack"], raw=False)
            if method["method"] == "AddNode":
                self.nodes.values[params["node_id"]] = properties
            elif method["method"] == "AddEdge":
                self.edges.append(
                    (params["source_id"], params["target_id"], properties)
                )
        version = envelope["content_version"]
        self.versions[version["object_id"]] = version
        self.records[envelope["envelope_id"]] = envelope
        return {
            "batch_id": mutation["batch_id"],
            "replayed": False,
            "projection_pending": False,
        }


class _FakeRdf:
    def validate_shacl(self, _shapes: str, _data_graph: str) -> dict[str, Any]:
        return {"conforms": True, "results": []}


class _FakeClient:
    def __init__(self) -> None:
        self.nodes = _FakeNodes()
        self.changes = _FakeChanges(self.nodes)
        self.rdf = _FakeRdf()

    @staticmethod
    def supports(operation: str) -> bool:
        return operation == "ApplyChangeEnvelope"


def test_ingest_entities_writes_nodes_and_edges():
    c = _FakeClient()
    res = ingest_entities(
        [
            {"id": "a", "node_type": "Repository", "name": "r"},
            {"id": "b", "node_type": "Organization"},
        ],
        [{"source": "a", "target": "b", "relationship": "ownedByOrg"}],
        client=c,
    )
    assert res == {"nodes": 2, "edges": 1}
    assert len(c.changes.applied) == 1
    assert set(c.nodes.values) == {"a", "b"}
    # provenance is stamped
    assert c.nodes.values["a"]["source"] == "github-agent"
    assert c.nodes.values["a"]["domain"] == "github"
    assert c.changes.edges == [("a", "b", {"relationship": "ownedByOrg"})]


def test_ingest_repositories_maps_repo_and_owner():
    c = _FakeClient()
    res = ingest_repositories(
        [
            {
                "id": 42,
                "name": "api",
                "full_name": "acme/api",
                "html_url": "https://github.com/acme/api",
                "default_branch": "main",
                "private": True,
                "owner": {
                    "login": "acme",
                    "id": 7,
                    "type": "Organization",
                    "html_url": "https://github.com/acme",
                },
            }
        ],
        client=c,
    )
    assert res == {"nodes": 2, "edges": 1}
    repo = c.nodes.values["github:repository:42"]
    assert repo["node_type"] == "Repository"
    assert repo["fullName"] == "acme/api"
    assert repo["isPrivate"] is True
    assert repo["externalToolId"] == "42"
    org = c.nodes.values["github:organization:acme"]
    assert org["node_type"] == "Organization"
    assert c.changes.edges == [
        (
            "github:repository:42",
            "github:organization:acme",
            {"relationship": "ownedByOrg"},
        )
    ]


def test_ingest_repositories_user_owner_is_person():
    c = _FakeClient()
    ingest_repositories(
        [
            {
                "id": 1,
                "name": "dotfiles",
                "owner": {"login": "octocat", "id": 9, "type": "User"},
            }
        ],
        client=c,
    )
    assert c.nodes.values["github:organization:octocat"]["node_type"] == "Person"


def test_ingest_pull_requests_maps_author_and_repo_link():
    c = _FakeClient()
    res = ingest_pull_requests(
        [
            {
                "id": 100,
                "number": 5,
                "title": "Add backoff",
                "state": "open",
                "draft": False,
                "user": {"login": "octocat", "id": 9},
            }
        ],
        repo_node_id="github:repository:42",
        client=c,
    )
    assert res == {"nodes": 2, "edges": 2}
    pr = c.nodes.values["github:pullrequest:100"]
    assert pr["node_type"] == "PullRequest"
    assert pr["number"] == 5
    assert c.nodes.values["github:person:octocat"]["node_type"] == "Person"
    assert (
        "github:pullrequest:100",
        "github:repository:42",
        {"relationship": "belongsToRepository"},
    ) in c.changes.edges


def test_ingest_issues_skips_pull_requests():
    c = _FakeClient()
    res = ingest_issues(
        [
            {"id": 1, "number": 1, "title": "bug", "state": "open"},
            {"id": 2, "number": 2, "title": "pr-in-disguise", "pull_request": {}},
        ],
        repo_node_id="github:repository:42",
        client=c,
    )
    # Only the real issue is written (+ its repo edge).
    assert res == {"nodes": 1, "edges": 1}
    assert "github:issue:1" in c.nodes.values
    assert "github:issue:2" not in c.nodes.values


def test_ingest_release_notes_writes_documents():
    c = _FakeClient()
    res = ingest_release_notes(
        [
            {
                "id": 500,
                "tag_name": "v1.0.0",
                "name": "First",
                "body": "## Highlights\n- ships it",
                "html_url": "https://github.com/acme/api/releases/tag/v1.0.0",
            },
            {"id": 501, "tag_name": "v1.0.1", "body": ""},
        ],
        repo_full_name="acme/api",
        client=c,
    )
    # Empty-body release is skipped.
    assert res == {"nodes": 1, "edges": 0}
    doc = c.nodes.values["github:release:500"]
    assert doc["node_type"] == "Document"
    assert doc["text"].startswith("## Highlights")
    # native_ingest's governed PII scrubber redacts uri-shaped values.
    assert doc["source_uri"] == "[REDACTED_LOCATION]"
    assert doc["source"] == "github-agent"


def test_ingest_pipeline_runs_maps_run_repo_commit_pr_and_jobs():
    c = _FakeClient()
    res = ingest_pipeline_runs(
        [
            {
                "id": 555,
                "status": "completed",
                "conclusion": "success",
                "head_sha": "abc123",
                "head_branch": "main",
                "event": "push",
                "html_url": "https://github.com/acme/api/actions/runs/555",
                "run_started_at": "2026-07-10T10:00:00Z",
                "updated_at": "2026-07-10T10:05:00Z",
                "pull_requests": [{"id": 100, "number": 5}],
            }
        ],
        repo_full_name="acme/api",
        repo_node_id="github:repository:42",
        jobs_by_run={
            555: [
                {
                    "id": 9001,
                    "name": "build",
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2026-07-10T10:00:05Z",
                    "completed_at": "2026-07-10T10:04:55Z",
                    "html_url": "https://github.com/acme/api/actions/runs/555/job/9001",
                }
            ]
        },
        client=c,
    )
    # PipelineRun + Commit + CheckRun nodes; ranFor(repo)+ranFor(commit)+ranFor(PR)+hasJob edges.
    assert res == {"nodes": 3, "edges": 4}

    run = c.nodes.values["github:pipelinerun:acme/api:555"]
    assert run["node_type"] == "PipelineRun"
    assert run["status"] == "completed"
    assert run["conclusion"] == "success"
    assert run["headSha"] == "abc123"
    assert run["headBranch"] == "main"
    assert run["event"] == "push"
    assert run["durationSeconds"] == 300
    assert run["externalToolId"] == "555"

    commit = c.nodes.values["github:commit:acme/api:abc123"]
    assert commit["node_type"] == "Commit"
    assert commit["sha"] == "abc123"

    job = c.nodes.values["github:checkrun:acme/api:9001"]
    assert job["node_type"] == "CheckRun"
    assert job["name"] == "build"
    assert job["status"] == "completed"
    assert job["conclusion"] == "success"

    assert (
        "github:pipelinerun:acme/api:555",
        "github:repository:42",
        {"relationship": "ranFor"},
    ) in c.changes.edges
    assert (
        "github:pipelinerun:acme/api:555",
        "github:commit:acme/api:abc123",
        {"relationship": "ranFor"},
    ) in c.changes.edges
    assert (
        "github:pipelinerun:acme/api:555",
        "github:pullrequest:100",
        {"relationship": "ranFor"},
    ) in c.changes.edges
    assert (
        "github:pipelinerun:acme/api:555",
        "github:checkrun:acme/api:9001",
        {"relationship": "hasJob"},
    ) in c.changes.edges


def test_ingest_pipeline_runs_minimal_no_links():
    c = _FakeClient()
    res = ingest_pipeline_runs(
        [{"id": 1, "status": "in_progress"}],
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    run = c.nodes.values["github:pipelinerun:None:1"]
    assert run["status"] == "in_progress"
    assert "durationSeconds" not in run


def test_ingest_noops_without_engine():
    # No injected client + no reachable engine -> clean no-op.
    assert ingest_entities([{"id": "a", "node_type": "Repository"}]) is None


def test_ingest_empty_is_noop():
    assert ingest_entities([], client=_FakeClient()) is None
    assert ingest_repositories([], client=_FakeClient()) is None
    assert ingest_release_notes([], client=_FakeClient()) is None
    assert ingest_pipeline_runs([], client=_FakeClient()) is None
