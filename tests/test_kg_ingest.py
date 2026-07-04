"""Native epistemic-graph typed-node + document ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` / ``ingest_documents`` seam and the GitHub
record mappers with a fake engine client (no engine required), asserting the txn
add_node/commit + edge calls and the repository/PR/issue/release mappings.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from github_agent.kg_ingest import (
    ingest_entities,
    ingest_issues,
    ingest_pull_requests,
    ingest_release_notes,
    ingest_repositories,
)


class _FakeTxn:
    def __init__(self):
        self.nodes = {}
        self.committed = False

    def begin(self, graph=None):
        self.graph = graph
        return "txn-1"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = props

    def commit(self, txn):
        self.committed = True
        return True


class _FakeEdges:
    def __init__(self):
        self.edges = []

    def add(self, src, dst, props):
        self.edges.append((src, dst, props))


class _FakeClient:
    def __init__(self):
        self.txn = _FakeTxn()
        self.edges = _FakeEdges()


def test_ingest_entities_writes_nodes_and_edges():
    c = _FakeClient()
    res = ingest_entities(
        [
            {"id": "a", "type": "Repository", "name": "r"},
            {"id": "b", "type": "Organization"},
        ],
        [{"source": "a", "target": "b", "type": "ownedByOrg"}],
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 1}
    assert c.txn.committed is True
    assert set(c.txn.nodes) == {"a", "b"}
    # provenance is stamped
    assert c.txn.nodes["a"]["source"] == "github-agent"
    assert c.txn.nodes["a"]["domain"] == "github"
    assert c.edges.edges == [("a", "b", {"type": "ownedByOrg"})]


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
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 1}
    repo = c.txn.nodes["github:repository:42"]
    assert repo["type"] == "Repository"
    assert repo["fullName"] == "acme/api"
    assert repo["isPrivate"] is True
    assert repo["externalToolId"] == "42"
    org = c.txn.nodes["github:organization:acme"]
    assert org["type"] == "Organization"
    assert c.edges.edges == [
        ("github:repository:42", "github:organization:acme", {"type": "ownedByOrg"})
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
        graph="__commons__",
    )
    assert c.txn.nodes["github:organization:octocat"]["type"] == "Person"


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
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 2}
    pr = c.txn.nodes["github:pullrequest:100"]
    assert pr["type"] == "PullRequest"
    assert pr["number"] == 5
    assert c.txn.nodes["github:person:octocat"]["type"] == "Person"
    assert (
        "github:pullrequest:100",
        "github:repository:42",
        {"type": "belongsToRepository"},
    ) in c.edges.edges


def test_ingest_issues_skips_pull_requests():
    c = _FakeClient()
    res = ingest_issues(
        [
            {"id": 1, "number": 1, "title": "bug", "state": "open"},
            {"id": 2, "number": 2, "title": "pr-in-disguise", "pull_request": {}},
        ],
        repo_node_id="github:repository:42",
        client=c,
        graph="__commons__",
    )
    # Only the real issue is written (+ its repo edge).
    assert res == {"nodes": 1, "edges": 1}
    assert "github:issue:1" in c.txn.nodes
    assert "github:issue:2" not in c.txn.nodes


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
        graph="__commons__",
    )
    # Empty-body release is skipped.
    assert res == {"nodes": 1, "edges": 0}
    doc = c.txn.nodes["github:release:500"]
    assert doc["type"] == "Document"
    assert doc["text"].startswith("## Highlights")
    assert doc["source_uri"].endswith("v1.0.0")
    assert doc["source"] == "github-agent"


def test_ingest_noops_without_engine():
    # No injected client + no reachable engine -> clean no-op.
    assert ingest_entities([{"id": "a", "type": "Repository"}]) is None


def test_ingest_empty_is_noop():
    assert ingest_entities([], client=_FakeClient()) is None
    assert ingest_repositories([], client=_FakeClient()) is None
    assert ingest_release_notes([], client=_FakeClient()) is None
