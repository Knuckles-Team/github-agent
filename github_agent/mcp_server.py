#!/usr/bin/python
import warnings

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from fastmcp.utilities.logging import get_logger
from pydantic import Field

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

# General urllib3/chardet mismatch warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")

import logging
import sys
from pathlib import Path
from typing import Any

from scion_sdk.config import load_config
from scion_sdk.connector_identity import connector_version
from scion_sdk.mcp.concurrency import run_blocking
from scion_sdk.mcp.content import ConnectorContent
from scion_sdk.mcp.server import create_mcp_server
from scion_sdk.mcp.tool_surface import register_tool_surface

from github_agent.api_client import Api
from github_agent.auth import (
    get_client,
    get_graphql_client,
)

__version__ = connector_version("github_agent")
logger = get_logger("GithubMCPServer")
logger.setLevel(logging.INFO)


_PIPELINE_LIST_KEYS = ("owner", "repo", "status", "branch", "per_page", "max_pages")


def _ingest_pipeline_list_kwargs(kwargs: dict) -> dict:
    """Keep only the github_get_workflow_runs filter keys, dropping None values."""
    return {
        k: v for k, v in kwargs.items() if k in _PIPELINE_LIST_KEYS and v is not None
    }


def _ingest_pipeline_normalize_runs(records: list) -> list[dict]:
    """Coerce each API run record to a plain dict, dropping any None entries."""
    return [
        run.model_dump() if hasattr(run, "model_dump") else run
        for run in records
        if run is not None
    ]


async def _ingest_pipeline_jobs_by_run(
    client: Any, owner: str, repo: str, runs: list[dict], include_jobs: bool
) -> dict[int, Any]:
    """Best-effort fetch of each run's jobs, keyed by run id.

    Returns an empty dict immediately when `include_jobs` is false. A single
    run's jobs fetch failing is logged and skipped rather than aborting the
    whole ingest.
    """
    jobs_by_run: dict[int, Any] = {}
    if not include_jobs:
        return jobs_by_run
    for run in runs:
        run_id = run.get("id")
        if run_id is None:
            continue
        try:
            jobs_response = await run_blocking(
                client.get_workflow_run_jobs,
                owner=owner,
                repo=repo,
                run_id=run_id,
            )
            jobs_by_run[run_id] = jobs_response.data
        except Exception as e:
            logger.debug(
                "github_ingest_pipelines: jobs fetch failed for run %s: %s",
                run_id,
                e,
            )
    return jobs_by_run


def register_ingest_tools(mcp: FastMCP):
    @mcp.tool(tags={"kg"})
    async def github_ingest_repos(
        visibility: str | None = Field(
            default=None, description="Only repositories with this visibility."
        ),
        affiliation: str | None = Field(
            default=None,
            description="Comma-separated affiliations: owner, collaborator, organization_member.",
        ),
        type: str | None = Field(
            default=None,
            description="Repository type filter: all, owner, public, private, member.",
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> Any:
        """Natively ingest GitHub repositories into medulla as typed :Repository nodes.

        Lists the authenticated user's repositories via the GitHub API and pushes them
        (with their owner :Organization/:Person + :ownedByOrg links) into the knowledge
        graph via the fast engine client. Best-effort: returns ``{"ingested": None}``
        when no engine is reachable. CONCEPT:AU-KG.ingest.enterprise-source-extractor.
        """
        if ctx:
            await ctx.info("Ingesting GitHub repositories into the knowledge graph...")
        from github_agent.kg_ingest import ingest_repositories

        kwargs = {
            k: v
            for k, v in {
                "visibility": visibility,
                "affiliation": affiliation,
                "type": type,
            }.items()
            if v is not None
        }
        response = await run_blocking(client.get_repositories, **kwargs)
        records = [
            repo.model_dump() if hasattr(repo, "model_dump") else repo
            for repo in response.data
            if repo is not None
        ]
        result = await ingest_repositories(records)
        return {"listed": len(records), "ingested": result}

    @mcp.tool(tags={"kg"})
    async def github_ingest_pipelines(
        owner: str = Field(description="Repository owner."),
        repo: str = Field(description="Repository name."),
        status: str | None = Field(
            default=None, description="Only runs with this status."
        ),
        branch: str | None = Field(
            default=None, description="Only runs on this branch."
        ),
        per_page: int | None = Field(default=None, description="Runs per page."),
        max_pages: int | None = Field(
            default=None, description="Maximum pages to read."
        ),
        include_jobs: bool = Field(
            default=True,
            description="Also fetch each run's jobs as :CheckRun children.",
        ),
        repo_node_id: str | None = Field(
            default=None,
            description="The :Repository node id to link runs to via ranFor.",
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> Any:
        """Natively ingest GitHub Actions workflow runs into medulla.

        Lists recent workflow runs for a repository (+ each run's jobs, unless
        ``include_jobs=false``) via the GitHub API and pushes them as typed
        :PipelineRun nodes (with child :CheckRun jobs linked via hasJob, and
        ranFor links to the repo/head-commit/PR) into the knowledge graph via
        the fast engine client. Uses the SAME :PipelineRun/:CheckRun classes as
        gitlab-api's CI ingestion so GitHub Actions and GitLab CI/CD unify under
        one node shape. Best-effort: returns ``{"ingested": None}`` when no
        engine is reachable. CONCEPT:AU-KG.ingest.enterprise-source-extractor.
        """
        if ctx:
            await ctx.info(
                "Ingesting GitHub Actions pipeline runs into the knowledge graph..."
            )
        from github_agent.kg_ingest import ingest_pipeline_runs

        list_kwargs = _ingest_pipeline_list_kwargs(
            {
                "owner": owner,
                "repo": repo,
                "status": status,
                "branch": branch,
                "per_page": per_page,
                "max_pages": max_pages,
            }
        )

        response = await run_blocking(client.get_workflow_runs, **list_kwargs)
        runs = _ingest_pipeline_normalize_runs(response.data)

        jobs_by_run = await _ingest_pipeline_jobs_by_run(
            client, owner, repo, runs, include_jobs
        )

        result = await ingest_pipeline_runs(
            runs,
            repo_full_name=f"{owner}/{repo}",
            repo_node_id=repo_node_id,
            jobs_by_run=jobs_by_run,
        )
        return {"listed": len(runs), "ingested": result}


def register_auto_merge_tools(mcp: FastMCP):
    """Register the GraphQL-only pull request auto-merge tools."""

    @mcp.tool(
        tags={"pulls"}, annotations={"readOnlyHint": False, "destructiveHint": False}
    )
    async def github_pulls_enable_auto_merge(
        owner: str = Field(description="Repository owner."),
        repo: str = Field(description="Repository name."),
        number: int = Field(description="Pull request number."),
        merge_method: str = Field(
            default="MERGE", description="MERGE, SQUASH or REBASE."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(default=None, description="MCP context"),
    ) -> Any:
        """Turn on auto-merge for a pull request: it merges once required checks pass."""
        try:
            gql_client = await run_blocking(get_graphql_client)
        except Exception as e:
            return {
                "status": 500,
                "error": f"GraphQL client unavailable: {type(e).__name__}",
                "data": None,
            }
        pr = await run_blocking(
            client.get_pull_request, owner=owner, repo=repo, number=number
        )
        data = await run_blocking(
            gql_client.enable_pull_request_auto_merge,
            pull_request_id=pr.data.node_id,
            merge_method=merge_method,
        )
        return {
            "status": 200,
            "message": "Auto-merge enabled successfully",
            "data": data,
        }

    @mcp.tool(
        tags={"pulls"}, annotations={"readOnlyHint": False, "destructiveHint": False}
    )
    async def github_pulls_disable_auto_merge(
        owner: str = Field(description="Repository owner."),
        repo: str = Field(description="Repository name."),
        number: int = Field(description="Pull request number."),
        client=Depends(get_client),
        ctx: Context | None = Field(default=None, description="MCP context"),
    ) -> Any:
        """Turn off auto-merge for a pull request."""
        try:
            gql_client = await run_blocking(get_graphql_client)
        except Exception as e:
            return {
                "status": 500,
                "error": f"GraphQL client unavailable: {type(e).__name__}",
                "data": None,
            }
        pr = await run_blocking(
            client.get_pull_request, owner=owner, repo=repo, number=number
        )
        data = await run_blocking(
            gql_client.disable_pull_request_auto_merge, pull_request_id=pr.data.node_id
        )
        return {
            "status": 200,
            "message": "Auto-merge disabled successfully",
            "data": data,
        }


def register_graphql_tools(mcp: FastMCP):
    from github_agent.auth import get_graphql_client

    @mcp.tool(tags={"graphql"})
    async def github_graphql(
        query: str = Field(
            description="The raw GraphQL query or mutation string to execute against the GitHub API."
        ),
        variables: str = Field(
            default="{}",
            description="JSON string of variables to pass along with the query.",
        ),
        operation_name: str | None = Field(
            default=None,
            description="Optional operation name if executing a specific query within the document.",
        ),
        client=Depends(get_graphql_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> Any:
        """Execute raw GraphQL queries and mutations natively on GitHub.

        One request can fan out across many repositories via aliased sub-queries
        (e.g. fetch the latest-commit CI status for the whole fleet in a single
        call) — far cheaper than one REST request per repository.
        """
        if ctx:
            await ctx.info("Executing GitHub GraphQL query...")
        import json

        try:
            vars_dict = json.loads(variables) if variables else None
        except Exception as e:
            return {"error": f"Invalid variables JSON: {type(e).__name__}"}

        try:
            return await run_blocking(
                client.execute_gql,
                query_str=query,
                variables=vars_dict,
                operation_name=operation_name,
            )
        except Exception as e:
            return {"error": f"GraphQL execution failed: {type(e).__name__}"}

    @mcp.tool(tags={"graphql"})
    async def github_discover_graphql_schema(
        type_name: str | None = Field(
            default=None,
            description="Optional GraphQL type name to inspect (e.g. 'Repository', 'CheckSuite'). If omitted, lists all available types.",
        ),
        client=Depends(get_graphql_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Discover the live GitHub GraphQL schema (types, fields, and attributes) in real-time."""
        from scion_sdk.mcp.graphql import (
            ctx_graphql_get_type_details as graphql_type_details,
        )
        from scion_sdk.mcp.graphql import (
            ctx_graphql_list_types as graphql_schema_types,
        )

        if ctx:
            await ctx.info("Retrieving live GitHub GraphQL schema...")

        async def execute_fn(q, variables=None):
            return await run_blocking(
                client.execute_gql, query_str=q, variables=variables
            )

        try:
            if type_name:
                return dict(await graphql_type_details(execute_fn, type_name))
            return dict(await graphql_schema_types(execute_fn))
        except Exception:
            return {"error": "Failed to discover GitHub GraphQL schema"}


TOOL_REGISTRY = [
    ("pulls", "PULLTOOL", register_auto_merge_tools),
    ("ingest", "INGESTTOOL", register_ingest_tools),
    ("graphql", "GRAPHQLTOOL", register_graphql_tools),
]


def get_mcp_instance() -> tuple[Any, Any, Any, Any, Any]:
    """Build the server: one typed tool per GitHub API operation."""
    load_config()
    args, mcp, middlewares = create_mcp_server(
        name="Github MCP",
        version=__version__,
        instructions="Github MCP Server - Manage your repositories, issues, and pull requests.",
        content=ConnectorContent.for_package(
            "github_agent",
            Path(__file__).resolve().parent,
            __version__,
        ),
    )

    registered_tags = register_tool_surface(
        mcp,
        client_cls=Api,
        get_client=get_client,
        service="github-agent",
        tool_registry=TOOL_REGISTRY,
    )

    for mw in middlewares:
        mcp.add_middleware(mw)

    tools_dict = (
        mcp._tools
        if hasattr(mcp, "_tools")
        else mcp.get_tools()
        if hasattr(mcp, "get_tools")
        else {}
    )
    imported_tools = list(tools_dict.keys())
    return mcp, args, middlewares, registered_tags, imported_tools


def mcp_server() -> None:
    mcp, args, middlewares, registered_tags, imported_tools = get_mcp_instance()
    print(f"Starting GitHub Agent MCP v{__version__}", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        logger.error(f"Unsupported transport: {args.transport}")
        sys.exit(1)


if __name__ == "__main__":
    mcp_server()
