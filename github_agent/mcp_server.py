#!/usr/bin/python
import warnings

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

import os
import sys
from typing import Dict, Any

from fastmcp import FastMCP, Context
from pydantic import Field

from agent_utilities.base_utilities import to_boolean, get_logger
from agent_utilities.mcp_utilities import (
    create_mcp_server,
)
from github_agent.auth import get_client

__version__ = "1.0.0"

logger = get_logger("GithubMCPServer")


def register_repo_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "List Repositories",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"repos"},
    )
    async def github_list_repos(
        visibility: str = Field(None, description="all, public, or private"),
        type: str = Field(None, description="all, owner, public, private, member"),
        _ctx: Context = Field(None, description="MCP context"),
    ) -> Dict[str, Any]:
        """List repositories for the authenticated user."""
        client = get_client()
        try:
            response = client.get_repositories(visibility=visibility, type=type)
            return {
                "status": 200,
                "message": "Repositories retrieved successfully",
                "data": [repo.model_dump() for repo in response.data],
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error listing repos: {str(e)}")
            return {
                "status": 500,
                "message": "Failed to list repos",
                "data": None,
                "error": str(e),
            }

    @mcp.tool(
        annotations={
            "title": "Get Repository",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"repos"},
    )
    async def github_get_repo(
        owner: str = Field(..., description="Repository owner"),
        repo: str = Field(..., description="Repository name"),
        _ctx: Context = Field(None, description="MCP context"),
    ) -> Dict[str, Any]:
        """Get details for a specific repository."""
        client = get_client()
        try:
            response = client.get_repository(owner=owner, repo=repo)
            return {
                "status": 200,
                "message": "Repository retrieved successfully",
                "data": response.data.model_dump(),
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error getting repo: {str(e)}")
            return {
                "status": 500,
                "message": "Failed to get repo",
                "data": None,
                "error": str(e),
            }


def register_issue_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "List Issues",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"issues"},
    )
    async def github_list_issues(
        owner: str = Field(..., description="Repository owner"),
        repo: str = Field(..., description="Repository name"),
        state: str = Field(None, description="open, closed, or all"),
        labels: str = Field(None, description="Comma-separated list of labels"),
        _ctx: Context = Field(None, description="MCP context"),
    ) -> Dict[str, Any]:
        """List issues for a repository."""
        client = get_client()
        try:
            response = client.get_issues(
                owner=owner, repo=repo, state=state, labels=labels
            )
            return {
                "status": 200,
                "message": "Issues retrieved successfully",
                "data": [issue.model_dump() for issue in response.data],
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error listing issues: {str(e)}")
            return {
                "status": 500,
                "message": "Failed to list issues",
                "data": None,
                "error": str(e),
            }


def register_pull_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "List Pull Requests",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"pulls"},
    )
    async def github_list_pull_requests(
        owner: str = Field(..., description="Repository owner"),
        repo: str = Field(..., description="Repository name"),
        state: str = Field(None, description="open, closed, or all"),
        _ctx: Context = Field(None, description="MCP context"),
    ) -> Dict[str, Any]:
        """List pull requests for a repository."""
        client = get_client()
        try:
            response = client.get_pull_requests(owner=owner, repo=repo, state=state)
            return {
                "status": 200,
                "message": "Pull requests retrieved successfully",
                "data": [pr.model_dump() for pr in response.data],
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error listing PRs: {str(e)}")
            return {
                "status": 500,
                "message": "Failed to list PRs",
                "data": None,
                "error": str(e),
            }


def register_content_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Get Contents",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"contents"},
    )
    async def github_get_contents(
        owner: str = Field(..., description="Repository owner"),
        repo: str = Field(..., description="Repository name"),
        path: str = Field(..., description="File or directory path"),
        ref: str = Field(None, description="Branch/Tag/Commit SHA"),
        _ctx: Context = Field(None, description="MCP context"),
    ) -> Dict[str, Any]:
        """Get contents of a file or directory."""
        client = get_client()
        try:
            response = client.get_contents(owner=owner, repo=repo, path=path, ref=ref)
            if isinstance(response.data, list):
                data = [item.model_dump() for item in response.data]
            else:
                data = response.data.model_dump()
            return {
                "status": 200,
                "message": "Contents retrieved successfully",
                "data": data,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error getting contents: {str(e)}")
            return {
                "status": 500,
                "message": "Failed to get contents",
                "data": None,
                "error": str(e),
            }


def get_mcp_instance() -> tuple[Any, Any, Any, Any, Any]:
    args, mcp, middlewares = create_mcp_server(
        name="Github MCP",
        version=__version__,
        instructions="Github MCP Server - Manage your repositories, issues, and pull requests.",
    )

    REPOSTOOL = to_boolean(os.getenv("REPOSTOOL", "True"))
    if REPOSTOOL:
        register_repo_tools(mcp)

    ISSUETOOL = to_boolean(os.getenv("ISSUETOOL", "True"))
    if ISSUETOOL:
        register_issue_tools(mcp)

    PULLSTOOL = to_boolean(os.getenv("PULLSTOOL", "True"))
    if PULLSTOOL:
        register_pull_tools(mcp)

    CONTENTSTOOL = to_boolean(os.getenv("CONTENTSTOOL", "True"))
    if CONTENTSTOOL:
        register_content_tools(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    registered_tags = []
    # FastMCP typically stores tools in .get_tools() or ._tools
    tools_dict = (
        mcp._tools
        if hasattr(mcp, "_tools")
        else mcp.get_tools() if hasattr(mcp, "get_tools") else {}
    )
    for tool in tools_dict.values():
        if hasattr(tool, "tags"):
            registered_tags.extend(list(tool.tags))

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
