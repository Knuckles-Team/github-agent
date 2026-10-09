"""Native epistemic-graph ingestion for GitHub records (typed graph nodes + documents).

CONCEPT:AU-KG.ingest.enterprise-source-extractor. The github-agent connector natively
pushes its data into the ONE epistemic-graph knowledge graph as **typed OWL nodes**
(``:Repository``, ``:PullRequest``, ``:Issue``, ``:Release``, ``:Organization``,
``:Person``, ``:PipelineRun``, ``:CheckRun``) plus links, and release notes as
**:Document** nodes for semantic search, through the ``agent_connector_sdk.ingest``
facade — the one connector write path; there is no self-contained fallback
transaction here.

This is a thin mapper: entities use the canonical ``node_type`` field and
relationships the canonical ``relationship`` field. ``ingest_entities`` /
``ingest_documents`` are best-effort: they return ``None`` (never raise) for empty
input or when the SDK reports :class:`IngestError`/:class:`IngestUnavailableError`
(no reachable engine, no verified session, or a malformed record). Node ids follow
``github:<class>:<externalId>``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agent_connector_sdk.ingest import (
    ChangeSet,
    Entity,
    IngestBinding,
    IngestError,
    IngestUnavailableError,
    KnowledgeIngest,
    Relationship,
    current_ingest,
)

logger = logging.getLogger("github_agent.kg")

_ENTITY_BINDING = IngestBinding(connector="github-agent", stream="github")
_DOCUMENT_BINDING = IngestBinding(connector="github-agent", stream="github-documents")


def _to_entity(record: dict[str, Any]) -> Entity:
    return Entity(
        id=record.get("id"),
        node_type=record.get("node_type"),
        properties={k: v for k, v in record.items() if k not in ("id", "node_type")},
    )


def _to_relationship(record: dict[str, Any]) -> Relationship:
    props = {
        k: v
        for k, v in record.items()
        if k not in ("source", "target", "relationship")
    }
    return Relationship(
        source=record["source"],
        target=record["target"],
        relationship=record["relationship"],
        properties=props or None,
    )


async def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    ingest: KnowledgeIngest | None = None,
) -> dict[str, int] | None:
    """Write typed OWL nodes (+ edges) into epistemic-graph. Best-effort, never raises.

    ``entities``: ``[{"id":..., "node_type":<owl:Class>, ...props}]``.
    ``relationships``: ``[{"source":id, "target":id, "relationship":<link>}]``.
    Returns ``{"nodes":n, "edges":m}`` or ``None`` (empty input / no reachable engine /
    malformed record). ``ingest`` may be injected (tests); otherwise the
    process-installed knowledge-ingest service is resolved on demand.
    """
    entities = [e for e in (entities or []) if e.get("id")]
    if not entities:
        return None
    change_set = ChangeSet(
        entities=tuple(_to_entity(e) for e in entities),
        relationships=tuple(_to_relationship(r) for r in relationships or ()),
    )
    try:
        service = ingest if ingest is not None else current_ingest()
        receipt = await service.submit(_ENTITY_BINDING, change_set)
        return {"nodes": receipt.affected_count, "edges": receipt.relationship_count}
    except (IngestError, IngestUnavailableError) as exc:
        logger.debug("KG ingest unavailable/failed: %s", exc)
        return None


def _document_node(doc: dict[str, Any], now: str) -> dict[str, Any] | None:
    """Map one raw document dict to a ``:Document`` node, or ``None`` if unmappable."""
    did = doc.get("id")
    text = doc.get("text") or doc.get("content")
    if not did or not text:
        return None
    node = {k: v for k, v in doc.items() if k != "content" and v is not None}
    node["id"] = did
    node["node_type"] = "Document"
    node["text"] = text
    node.setdefault("created_at", now)
    return node


async def ingest_documents(
    documents: list[dict[str, Any]],
    *,
    ingest: KnowledgeIngest | None = None,
) -> dict[str, int] | None:
    """Write text records as ``:Document`` nodes (semantic-search fodder). Best-effort.

    Each doc: ``{"id":..., "text":..., "title"?:..., "source_uri"?:..., ...props}``.
    Returns ``{"nodes":n, "edges":0}`` or ``None``.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nodes = [
        node
        for doc in (documents or [])
        if (node := _document_node(doc, now)) is not None
    ]
    if not nodes:
        return None
    change_set = ChangeSet(entities=tuple(_to_entity(n) for n in nodes))
    try:
        service = ingest if ingest is not None else current_ingest()
        receipt = await service.submit(_DOCUMENT_BINDING, change_set)
        return {"nodes": receipt.affected_count, "edges": receipt.relationship_count}
    except (IngestError, IngestUnavailableError) as exc:
        logger.debug("KG ingest unavailable/failed: %s", exc)
        return None


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
        "node_type": "Person",
        "name": login,
        "htmlUrl": _str(user.get("html_url")),
        "externalToolId": str(user.get("id")) if user.get("id") is not None else login,
    }


def _str(value: Any) -> Any:
    """Coerce pydantic HttpUrl / non-JSON scalars to ``str`` (leave None as None)."""
    return None if value is None else str(value)


async def ingest_repositories(
    repositories: list[dict[str, Any]],
    *,
    ingest: KnowledgeIngest | None = None,
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
                "node_type": "Repository",
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
                    "node_type": "Organization"
                    if owner_type == "organization"
                    else "Person",
                    "name": login,
                    "htmlUrl": _str(owner.get("html_url")),
                }
            )
            relationships.append(
                {"source": node_id, "target": oid, "relationship": "ownedByOrg"}
            )
    return await ingest_entities(entities, relationships, ingest=ingest)


async def ingest_pull_requests(
    pull_requests: list[dict[str, Any]],
    *,
    repo_node_id: str | None = None,
    ingest: KnowledgeIngest | None = None,
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
                "node_type": "PullRequest",
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
                {"source": node_id, "target": author_id, "relationship": "authoredBy"}
            )
        if repo_node_id:
            relationships.append(
                {
                    "source": node_id,
                    "target": repo_node_id,
                    "relationship": "belongsToRepository",
                }
            )
    return await ingest_entities(entities, relationships, ingest=ingest)


async def ingest_issues(
    issues: list[dict[str, Any]],
    *,
    repo_node_id: str | None = None,
    ingest: KnowledgeIngest | None = None,
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
                "node_type": "Issue",
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
                {"source": node_id, "target": author_id, "relationship": "authoredBy"}
            )
        if repo_node_id:
            relationships.append(
                {
                    "source": node_id,
                    "target": repo_node_id,
                    "relationship": "belongsToRepository",
                }
            )
    return await ingest_entities(entities, relationships, ingest=ingest)


async def ingest_release_notes(
    releases: list[dict[str, Any]],
    *,
    repo_full_name: str | None = None,
    ingest: KnowledgeIngest | None = None,
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
    return await ingest_documents(docs, ingest=ingest)


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


def _pipeline_run_entity(
    run: dict[str, Any], node_id: str, run_id: Any
) -> dict[str, Any]:
    """Map one workflow-run record to its ``:PipelineRun`` entity."""
    run_started = run.get("run_started_at")
    run_updated = run.get("updated_at")
    return {
        "id": node_id,
        "node_type": "PipelineRun",
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


def _commit_node_and_relationship(
    repo: str | None, head_sha: str | None, node_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | tuple[None, None]:
    """Map a run's head commit to a ``:Commit`` node + its ``ranFor`` edge, if known."""
    if not (repo and head_sha):
        return None, None
    commit_id = f"github:commit:{repo}:{head_sha}"
    commit = {
        "id": commit_id,
        "node_type": "Commit",
        "sha": head_sha,
        "externalToolId": head_sha,
    }
    relationship = {"source": node_id, "target": commit_id, "relationship": "ranFor"}
    return commit, relationship


def _pipeline_run_pr_relationships(
    run: dict[str, Any], node_id: str
) -> list[dict[str, Any]]:
    """``ranFor`` edges from a run to each pull request it lists."""
    return [
        {
            "source": node_id,
            "target": f"github:pullrequest:{pr['id']}",
            "relationship": "ranFor",
        }
        for pr in (run.get("pull_requests") or [])
        if pr.get("id") is not None
    ]


def _pipeline_run_job_entities_and_relationships(
    jobs: list[dict[str, Any]], repo: str | None, node_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map a run's jobs/check-runs to ``:CheckRun`` nodes + ``hasJob`` edges."""
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for job in jobs:
        job_id = job.get("id")
        if job_id is None:
            continue
        job_node_id = f"github:checkrun:{repo}:{job_id}"
        entities.append(
            {
                "id": job_node_id,
                "node_type": "CheckRun",
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
            {"source": node_id, "target": job_node_id, "relationship": "hasJob"}
        )
    return entities, relationships


def _pipeline_run_graph(
    run: dict[str, Any],
    repo_full_name: str | None,
    repo_node_id: str | None,
    jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Map one workflow run (+ its jobs) to its full entity/relationship set."""
    run_id = run.get("id")
    if run_id is None:
        return None
    repo = repo_full_name or (run.get("repository") or {}).get("full_name")
    node_id = f"github:pipelinerun:{repo}:{run_id}"

    entities = [_pipeline_run_entity(run, node_id, run_id)]
    relationships: list[dict[str, Any]] = []

    if repo_node_id:
        relationships.append(
            {"source": node_id, "target": repo_node_id, "relationship": "ranFor"}
        )

    commit, commit_relationship = _commit_node_and_relationship(
        repo, run.get("head_sha"), node_id
    )
    if commit is not None and commit_relationship is not None:
        entities.append(commit)
        relationships.append(commit_relationship)

    relationships.extend(_pipeline_run_pr_relationships(run, node_id))

    job_entities, job_relationships = _pipeline_run_job_entities_and_relationships(
        jobs, repo, node_id
    )
    entities.extend(job_entities)
    relationships.extend(job_relationships)

    return entities, relationships


async def ingest_pipeline_runs(
    runs: list[dict[str, Any]],
    *,
    repo_full_name: str | None = None,
    repo_node_id: str | None = None,
    jobs_by_run: dict[int, list[dict[str, Any]]] | None = None,
    ingest: KnowledgeIngest | None = None,
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
        jobs = jobs_by_run.get(run_id, []) if isinstance(run_id, int) else []
        run_graph = _pipeline_run_graph(run, repo_full_name, repo_node_id, jobs)
        if run_graph is None:
            continue
        run_entities, run_relationships = run_graph
        entities.extend(run_entities)
        relationships.extend(run_relationships)
    return await ingest_entities(entities, relationships, ingest=ingest)
