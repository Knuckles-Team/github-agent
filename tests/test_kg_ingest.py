"""Native epistemic-graph typed-node + document ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` / ``ingest_documents`` seam and the GitHub
record mappers against a fake SDK transport (no engine required), asserting the
submitted records/relationships and the repository/PR/issue/release mappings.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from agent_connector_sdk.ingest import KnowledgeIngest

from github_agent.kg_ingest import (
    ingest_entities,
    ingest_issues,
    ingest_pipeline_runs,
    ingest_pull_requests,
    ingest_release_notes,
    ingest_repositories,
)


class _FakeTransport:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def source_status(self, connector: str, stream: str) -> Any:
        return SimpleNamespace(accepted_checkpoint=None)

    async def submit(self, request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(
            affected_count=len(request.records),
            relationship_count=len(request.relationships),
        )

    async def store_blob(self, data: Any) -> Any:
        raise AssertionError("this connector's ingestion carries no media")


@pytest.fixture
def ingest():
    transport = _FakeTransport()
    return KnowledgeIngest(transport, loop=None), transport


def _records(transport: _FakeTransport) -> dict[str, Any]:
    return {r.record_id: r.payload for r in transport.requests[0].records}


def _edges(transport: _FakeTransport) -> set[tuple[str, str]]:
    return {
        (r.source.record_id, r.target.record_id)
        for r in transport.requests[0].relationships
    }


async def test_ingest_entities_writes_nodes_and_edges(ingest):
    service, transport = ingest
    res = await ingest_entities(
        [
            {"id": "a", "node_type": "Repository", "name": "r"},
            {"id": "b", "node_type": "Organization"},
        ],
        [{"source": "a", "target": "b", "relationship": "ownedByOrg"}],
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    records = _records(transport)
    assert set(records) == {"a", "b"}
    assert _edges(transport) == {("a", "b")}


async def test_ingest_repositories_maps_repo_and_owner(ingest):
    service, transport = ingest
    res = await ingest_repositories(
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
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    records = _records(transport)
    repo = records["github:repository:42"]
    assert repo["fullName"] == "acme/api"
    assert repo["isPrivate"] is True
    assert repo["externalToolId"] == "42"
    org = records["github:organization:acme"]
    assert org is not None
    assert _edges(transport) == {
        ("github:repository:42", "github:organization:acme")
    }


async def test_ingest_repositories_user_owner_is_person(ingest):
    service, transport = ingest
    await ingest_repositories(
        [
            {
                "id": 1,
                "name": "dotfiles",
                "owner": {"login": "octocat", "id": 9, "type": "User"},
            }
        ],
        ingest=service,
    )
    rel = transport.requests[0].relationships[0]
    assert rel.target.record_id == "github:organization:octocat"


async def test_ingest_pull_requests_maps_author_and_repo_link(ingest):
    service, transport = ingest
    res = await ingest_pull_requests(
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
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 2}
    records = _records(transport)
    pr = records["github:pullrequest:100"]
    assert pr["number"] == 5
    assert "github:person:octocat" in records
    assert (
        "github:pullrequest:100",
        "github:repository:42",
    ) in _edges(transport)


async def test_ingest_issues_skips_pull_requests(ingest):
    service, transport = ingest
    res = await ingest_issues(
        [
            {"id": 1, "number": 1, "title": "bug", "state": "open"},
            {"id": 2, "number": 2, "title": "pr-in-disguise", "pull_request": {}},
        ],
        repo_node_id="github:repository:42",
        ingest=service,
    )
    # Only the real issue is written (+ its repo edge).
    assert res == {"nodes": 1, "edges": 1}
    records = _records(transport)
    assert "github:issue:1" in records
    assert "github:issue:2" not in records


async def test_ingest_release_notes_writes_documents(ingest):
    service, transport = ingest
    res = await ingest_release_notes(
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
        ingest=service,
    )
    # Empty-body release is skipped.
    assert res == {"nodes": 1, "edges": 0}
    records = _records(transport)
    doc = records["github:release:500"]
    assert doc["text"].startswith("## Highlights")
    # the SDK's persistence-privacy guard redacts uri-shaped values.
    assert doc["source_uri"] == "[REDACTED_LOCATION]"


async def test_ingest_pipeline_runs_maps_run_repo_commit_pr_and_jobs(ingest):
    service, transport = ingest
    res = await ingest_pipeline_runs(
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
        ingest=service,
    )
    # PipelineRun + Commit + CheckRun nodes; ranFor(repo)+ranFor(commit)+ranFor(PR)+hasJob edges.
    assert res == {"nodes": 3, "edges": 4}

    records = _records(transport)
    run = records["github:pipelinerun:acme/api:555"]
    assert run["status"] == "completed"
    assert run["conclusion"] == "success"
    assert run["headSha"] == "abc123"
    assert run["headBranch"] == "main"
    assert run["event"] == "push"
    assert run["durationSeconds"] == 300
    assert run["externalToolId"] == "555"

    commit = records["github:commit:acme/api:abc123"]
    assert commit["sha"] == "abc123"

    job = records["github:checkrun:acme/api:9001"]
    assert job["name"] == "build"
    assert job["status"] == "completed"
    assert job["conclusion"] == "success"

    edges = _edges(transport)
    assert ("github:pipelinerun:acme/api:555", "github:repository:42") in edges
    assert (
        "github:pipelinerun:acme/api:555",
        "github:commit:acme/api:abc123",
    ) in edges
    assert (
        "github:pipelinerun:acme/api:555",
        "github:pullrequest:100",
    ) in edges
    assert (
        "github:pipelinerun:acme/api:555",
        "github:checkrun:acme/api:9001",
    ) in edges


async def test_ingest_pipeline_runs_minimal_no_links(ingest):
    service, transport = ingest
    res = await ingest_pipeline_runs(
        [{"id": 1, "status": "in_progress"}],
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 0}
    run = _records(transport)["github:pipelinerun:None:1"]
    assert run["status"] == "in_progress"
    assert "durationSeconds" not in run


async def test_ingest_noops_without_engine():
    # No injected ingest + no reachable engine -> clean no-op.
    assert await ingest_entities([{"id": "a", "node_type": "Repository"}]) is None


async def test_ingest_empty_is_noop(ingest):
    service, _transport = ingest
    assert await ingest_entities([], ingest=service) is None
    assert await ingest_repositories([], ingest=service) is None
    assert await ingest_release_notes([], ingest=service) is None
    assert await ingest_pipeline_runs([], ingest=service) is None
