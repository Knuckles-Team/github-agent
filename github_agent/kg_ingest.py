"""Native epistemic-graph ingestion for GitHub records (typed graph nodes + documents).

CONCEPT:AU-KG.ingest.enterprise-source-extractor. The github-agent connector natively
pushes its data into the ONE epistemic-graph knowledge graph as **typed OWL nodes**
(``:Repository``, ``:PullRequest``, ``:Issue``, ``:Release``, ``:Organization``,
``:Person``, ``:PipelineRun``, ``:CheckRun``) plus links, and release notes as
**:Document** nodes for semantic search —
using the lightweight engine client (``GraphComputeEngine()._client`` + ``txn``), the
same fast client the blob ``MediaStore`` uses, NOT the heavy in-process ingestion engine.

This is a thin mapper: it delegates the txn write path to the shared primitive
``agent_utilities.knowledge_graph.memory.native_ingest`` when available, and otherwise
falls back to a self-contained txn implementation (the primitive is not yet in every
installed agent_utilities). Entirely best-effort and dependency-/engine-guarded: with no
KG stack or no reachable engine every entry point **no-ops** (returns ``None``), so the
connector keeps working with zero KG infrastructure. Nodes carry the shared provenance
(``domain``/``source``) and match the classes federated by ``github_agent.ontology``.
Node ids follow ``github:<class>:<externalId>``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("github_agent.kg")

_SOURCE = "github-agent"
_DOMAIN = "github"
_DEFAULT_GRAPH = "__commons__"

# Prefer the shared native-ingest primitive; fall back to the self-contained txn path
# below when it is not present in the installed agent_utilities.
try:  # pragma: no cover - import availability varies by install
    from agent_utilities.knowledge_graph.memory.native_ingest import (
        ingest_documents as _shared_ingest_documents,
    )
    from agent_utilities.knowledge_graph.memory.native_ingest import (
        ingest_entities as _shared_ingest_entities,
    )
except Exception:  # noqa: BLE001 — primitive absent -> self-contained fallback
    _shared_ingest_entities = None
    _shared_ingest_documents = None


def _client() -> tuple[Any | None, str]:
    """Return ``(engine_client, graph_name)`` or ``(None, "")`` when unavailable."""
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )
    except Exception as e:  # noqa: BLE001 — KG stack absent
        logger.debug("KG ingest unavailable (import): %s", e)
        return None, ""
    try:
        engine = GraphComputeEngine()
        client = getattr(engine, "_client", None)
        if client is None:
            return None, ""
        return client, (getattr(engine, "graph_name", None) or _DEFAULT_GRAPH)
    except Exception as e:  # noqa: BLE001 — engine unreachable
        logger.debug("KG ingest: engine unreachable: %s", e)
        return None, ""


def _write_nodes(
    client: Any,
    graph: str,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None,
) -> dict[str, int] | None:
    """Stamp provenance, MERGE the nodes in one txn, then add the edges (fallback path)."""
    nodes = [n for n in nodes if n.get("id")]
    if not nodes:
        return None
    try:
        txn = client.txn.begin(graph=graph)
        for node in nodes:
            props = {k: v for k, v in node.items() if k != "id" and v is not None}
            props.setdefault("source", _SOURCE)
            props.setdefault("domain", _DOMAIN)
            client.txn.add_node(txn, node["id"], props)
        committed = client.txn.commit(txn)
    except Exception as e:  # noqa: BLE001 — engine/txn failure is non-fatal
        logger.warning("KG ingest: txn failed: %s", e)
        return None
    if not committed:
        logger.warning("KG ingest: txn not committed (conflict)")
        return None

    edges = 0
    for rel in relationships or []:
        try:
            client.edges.add(
                rel["source"], rel["target"], {"type": rel.get("type", "RELATED")}
            )
            edges += 1
        except Exception as e:  # noqa: BLE001 — pure edge link, best-effort
            logger.debug("KG ingest: edge skipped: %s", e)

    logger.info("KG ingest: wrote %d nodes, %d edges", len(nodes), edges)
    return {"nodes": len(nodes), "edges": edges}


def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write typed OWL nodes (+ edges) into epistemic-graph.

    ``entities``: ``[{"id":..., "type":<owl:Class>, ...props}]``.
    ``relationships``: ``[{"source":id, "target":id, "type":<link>}]``.
    Returns ``{"nodes":n, "edges":m}`` or ``None`` (no engine / failure; never raises).
    ``client``/``graph`` may be injected (tests); otherwise resolved on demand. When a
    client is injected the self-contained fallback is used so the write is observable.
    """
    entities = [e for e in (entities or []) if e.get("id")]
    if not entities:
        return None
    if client is None and _shared_ingest_entities is not None:
        return _shared_ingest_entities(
            entities, relationships, source=_SOURCE, domain=_DOMAIN
        )
    if client is None:
        client, graph = _client()
    if client is None:
        return None
    return _write_nodes(client, graph or _DEFAULT_GRAPH, entities, relationships)


def ingest_documents(
    documents: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write text records as ``:Document`` nodes (semantic-search fodder).

    Each doc: ``{"id":..., "text":..., "title"?:..., "source_uri"?:..., ...props}``.
    Returns ``{"nodes":n, "edges":0}`` or ``None``.
    """
    if client is None and _shared_ingest_documents is not None:
        return _shared_ingest_documents(documents, source=_SOURCE, domain=_DOMAIN)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nodes: list[dict[str, Any]] = []
    for doc in documents or []:
        did = doc.get("id")
        text = doc.get("text") or doc.get("content")
        if not did or not text:
            continue
        node = {k: v for k, v in doc.items() if k != "content" and v is not None}
        node["id"] = did
        node["type"] = "Document"
        node["text"] = text
        node.setdefault("created_at", now)
        nodes.append(node)
    if not nodes:
        return None
    if client is None:
        client, graph = _client()
    if client is None:
        return None
    return _write_nodes(client, graph or _DEFAULT_GRAPH, nodes, None)


def _person(user: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    """Map a GitHub user dict → ``(node_id, :Person entity)`` or ``(None, None)``."""
    if not user:
        return None, None
    login = user.get("login")
    if not login:
        return None, None
    pid = f"github:person:{login}"
    return pid, {
        "id": pid,
        "type": "Person",
        "name": login,
        "htmlUrl": _str(user.get("html_url")),
        "externalToolId": str(user.get("id")) if user.get("id") is not None else login,
    }


def _str(value: Any) -> Any:
    """Coerce pydantic HttpUrl / non-JSON scalars to ``str`` (leave None as None)."""
    return None if value is None else str(value)


def ingest_repositories(
    repositories: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map GitHub repository records → ``:Repository`` (+ owner ``:Organization``/``:Person``) nodes."""
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for repo in repositories or []:
        rid = repo.get("id")
        if rid is None:
            continue
        node_id = f"github:repository:{rid}"
        entities.append(
            {
                "id": node_id,
                "type": "Repository",
                "name": repo.get("name"),
                "fullName": repo.get("full_name"),
                "htmlUrl": _str(repo.get("html_url")),
                "defaultBranch": repo.get("default_branch"),
                "isPrivate": repo.get("private"),
                "language": repo.get("language"),
                "externalToolId": str(rid),
            }
        )
        owner = repo.get("owner") or {}
        login = owner.get("login")
        if login:
            owner_type = str(owner.get("type", "")).lower()
            oid = f"github:organization:{login}"
            entities.append(
                {
                    "id": oid,
                    "type": "Organization"
                    if owner_type == "organization"
                    else "Person",
                    "name": login,
                    "htmlUrl": _str(owner.get("html_url")),
                }
            )
            relationships.append(
                {"source": node_id, "target": oid, "type": "ownedByOrg"}
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_pull_requests(
    pull_requests: list[dict[str, Any]],
    *,
    repo_node_id: str | None = None,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map GitHub pull-request records → ``:PullRequest`` nodes (+ author, repo links)."""
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for pr in pull_requests or []:
        pid = pr.get("id")
        if pid is None:
            continue
        node_id = f"github:pullrequest:{pid}"
        entities.append(
            {
                "id": node_id,
                "type": "PullRequest",
                "title": pr.get("title"),
                "number": pr.get("number"),
                "state": pr.get("state"),
                "isDraft": pr.get("draft"),
                "htmlUrl": _str(pr.get("html_url")),
                "externalToolId": str(pid),
            }
        )
        author_id, author = _person(pr.get("user"))
        if author:
            entities.append(author)
            relationships.append(
                {"source": node_id, "target": author_id, "type": "authoredBy"}
            )
        if repo_node_id:
            relationships.append(
                {
                    "source": node_id,
                    "target": repo_node_id,
                    "type": "belongsToRepository",
                }
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_issues(
    issues: list[dict[str, Any]],
    *,
    repo_node_id: str | None = None,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map GitHub issue records → ``:Issue`` nodes (+ author, repo links).

    Skips items carrying a ``pull_request`` key (the /issues endpoint returns PRs too).
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for issue in issues or []:
        # GitHub's /issues endpoint returns PRs too; they carry a ``pull_request`` key.
        if "pull_request" in issue:
            continue
        iid = issue.get("id")
        if iid is None:
            continue
        node_id = f"github:issue:{iid}"
        entities.append(
            {
                "id": node_id,
                "type": "Issue",
                "title": issue.get("title"),
                "number": issue.get("number"),
                "state": issue.get("state"),
                "htmlUrl": _str(issue.get("html_url")),
                "externalToolId": str(iid),
            }
        )
        author_id, author = _person(issue.get("user"))
        if author:
            entities.append(author)
            relationships.append(
                {"source": node_id, "target": author_id, "type": "authoredBy"}
            )
        if repo_node_id:
            relationships.append(
                {
                    "source": node_id,
                    "target": repo_node_id,
                    "type": "belongsToRepository",
                }
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_release_notes(
    releases: list[dict[str, Any]],
    *,
    repo_full_name: str | None = None,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map GitHub release records → ``:Document`` nodes carrying the release-notes body."""
    docs: list[dict[str, Any]] = []
    for rel in releases or []:
        rid = rel.get("id")
        body = rel.get("body")
        if rid is None or not body:
            continue
        docs.append(
            {
                "id": f"github:release:{rid}",
                "text": body,
                "title": rel.get("name") or rel.get("tag_name"),
                "tagName": rel.get("tag_name"),
                "source_uri": _str(rel.get("html_url")),
                "repository": repo_full_name,
                "externalToolId": str(rid),
            }
        )
    return ingest_documents(docs, client=client, graph=graph)


def _duration_seconds(start: Any, end: Any) -> int | None:
    """Whole-second wall-clock duration between two ISO-8601 timestamps, or ``None``."""
    if not start or not end:
        return None
    try:
        import datetime as _dt

        started = _dt.datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        ended = _dt.datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return max(0, int((ended - started).total_seconds()))
    except (ValueError, TypeError):
        return None


def ingest_pipeline_runs(
    runs: list[dict[str, Any]],
    *,
    repo_full_name: str | None = None,
    repo_node_id: str | None = None,
    jobs_by_run: dict[int, list[dict[str, Any]]] | None = None,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map GitHub Actions workflow runs (+ jobs/check-runs) → ``:PipelineRun``/``:CheckRun``.

    ``runs``: raw ``WorkflowRun`` records (``client.get_workflow_runs`` /
    ``.get_workflow_run`` ``model_dump()``). ``jobs_by_run``: optional ``{run_id:
    [job_or_check_run, ...]}`` from ``client.get_workflow_run_jobs`` (or the Checks
    API) — each becomes a child ``:CheckRun`` linked via ``hasJob``.

    Uses the SAME ``:PipelineRun``/``:CheckRun`` classes and ``ranFor``/``hasJob``
    edge names as gitlab-api's ingestion so GitHub Actions and GitLab CI/CD unify
    under one CI node shape in the knowledge graph. ``ranFor`` is emitted once per
    known target — the repo, the head commit, and (if the run lists it) the PR.
    Stable ids: ``github:pipelinerun:<repo>:<id>`` / ``github:checkrun:<repo>:<id>``.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    jobs_by_run = jobs_by_run or {}
    for run in runs or []:
        run_id = run.get("id")
        if run_id is None:
            continue
        repo = repo_full_name or (run.get("repository") or {}).get("full_name")
        node_id = f"github:pipelinerun:{repo}:{run_id}"
        run_started = run.get("run_started_at")
        run_updated = run.get("updated_at")
        entities.append(
            {
                "id": node_id,
                "type": "PipelineRun",
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "headSha": run.get("head_sha"),
                "headBranch": run.get("head_branch"),
                "event": run.get("event"),
                "htmlUrl": _str(run.get("html_url")),
                "runStartedAt": run_started,
                "runUpdatedAt": run_updated,
                "durationSeconds": _duration_seconds(run_started, run_updated),
                "externalToolId": str(run_id),
            }
        )

        if repo_node_id:
            relationships.append(
                {"source": node_id, "target": repo_node_id, "type": "ranFor"}
            )

        head_sha = run.get("head_sha")
        if repo and head_sha:
            commit_id = f"github:commit:{repo}:{head_sha}"
            entities.append(
                {
                    "id": commit_id,
                    "type": "Commit",
                    "sha": head_sha,
                    "externalToolId": head_sha,
                }
            )
            relationships.append(
                {"source": node_id, "target": commit_id, "type": "ranFor"}
            )

        for pr in run.get("pull_requests") or []:
            pr_id = pr.get("id")
            if pr_id is None:
                continue
            relationships.append(
                {
                    "source": node_id,
                    "target": f"github:pullrequest:{pr_id}",
                    "type": "ranFor",
                }
            )

        for job in jobs_by_run.get(run_id) or []:
            job_id = job.get("id")
            if job_id is None:
                continue
            job_node_id = f"github:checkrun:{repo}:{job_id}"
            entities.append(
                {
                    "id": job_node_id,
                    "type": "CheckRun",
                    "name": job.get("name"),
                    "status": job.get("status"),
                    "conclusion": job.get("conclusion"),
                    "startedAt": job.get("started_at"),
                    "completedAt": job.get("completed_at"),
                    "htmlUrl": _str(job.get("html_url")),
                    "externalToolId": str(job_id),
                }
            )
            relationships.append(
                {"source": node_id, "target": job_node_id, "type": "hasJob"}
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)
