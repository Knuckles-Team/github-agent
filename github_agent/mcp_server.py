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
import os
import sys
from typing import Any

from agent_utilities.base_utilities import to_boolean
from agent_utilities.mcp_utilities import (
    create_mcp_server,
)

from github_agent.auth import get_client

__version__ = "1.0.0"
logger = get_logger("GithubMCPServer")
logger.setLevel(logging.INFO)


def register_repo_tools(mcp: FastMCP):
    @mcp.tool(tags={"repos"})
    async def github_repos(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get', 'create', 'delete', 'update'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub repositories."""
        if ctx:
            await ctx.info("Executing github_repos action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "list":
                response = client.get_repositories(**kwargs)
                return {
                    "status": 200,
                    "message": "Repositories retrieved successfully",
                    "data": [repo.model_dump() for repo in response.data],
                }
            elif action == "get":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = client.get_repository(owner=owner, repo=repo)
                return {
                    "status": 200,
                    "message": "Repository retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "create":
                name = kwargs.pop("name", None)
                if not name:
                    return {
                        "status": 400,
                        "error": "Missing required 'name' parameter",
                        "data": None,
                    }
                response = client.create_repository(name=name, **kwargs)
                return {
                    "status": 201,
                    "message": "Repository created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "delete":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = client.delete_repository(owner=owner, repo=repo)
                return {
                    "status": 200,
                    "message": "Repository deleted successfully",
                    "data": response.data,
                }
            elif action == "update":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = client.update_repository(owner=owner, repo=repo, **kwargs)
                return {
                    "status": 200,
                    "message": "Repository updated successfully",
                    "data": response.data.model_dump(),
                }
            else:
                return {
                    "status": 400,
                    "error": f"Unknown action: {action}",
                    "data": None,
                }
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}


def register_issue_tools(mcp: FastMCP):
    @mcp.tool(tags={"issues"})
    async def github_issues(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get', 'create', 'update'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub issues."""
        if ctx:
            await ctx.info("Executing github_issues action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "list":
                response = client.get_issues(**kwargs)
                return {
                    "status": 200,
                    "message": "Issues retrieved successfully",
                    "data": [issue.model_dump() for issue in response.data],
                }
            elif action == "get":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                number = kwargs.get("number")
                if not owner or not repo or not number:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'number' parameter",
                        "data": None,
                    }
                response = client.get_issue(owner=owner, repo=repo, number=int(number))
                return {
                    "status": 200,
                    "message": "Issue retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "create":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                title = kwargs.pop("title", None)
                if not owner or not repo or not title:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'title' parameter",
                        "data": None,
                    }
                response = client.create_issue(
                    owner=owner, repo=repo, title=title, **kwargs
                )
                return {
                    "status": 201,
                    "message": "Issue created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "update":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                number = kwargs.pop("number", None)
                if not owner or not repo or not number:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'number' parameter",
                        "data": None,
                    }
                response = client.update_issue(
                    owner=owner, repo=repo, number=int(number), **kwargs
                )
                return {
                    "status": 200,
                    "message": "Issue updated successfully",
                    "data": response.data.model_dump(),
                }
            else:
                return {
                    "status": 400,
                    "error": f"Unknown action: {action}",
                    "data": None,
                }
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}


def register_pull_tools(mcp: FastMCP):
    @mcp.tool(tags={"pulls"})
    async def github_pulls(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get', 'create', 'update'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub pull requests."""
        if ctx:
            await ctx.info("Executing github_pulls action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "list":
                response = client.get_pull_requests(**kwargs)
                return {
                    "status": 200,
                    "message": "Pull requests retrieved successfully",
                    "data": [pr.model_dump() for pr in response.data],
                }
            elif action == "get":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                number = kwargs.get("number")
                if not owner or not repo or not number:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'number' parameter",
                        "data": None,
                    }
                response = client.get_pull_request(
                    owner=owner, repo=repo, number=int(number)
                )
                return {
                    "status": 200,
                    "message": "Pull request retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "create":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                title = kwargs.pop("title", None)
                head = kwargs.pop("head", None)
                base = kwargs.pop("base", None)
                if not owner or not repo or not title or not head or not base:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', 'title', 'head', or 'base' parameter",
                        "data": None,
                    }
                response = client.create_pull_request(
                    owner=owner, repo=repo, title=title, head=head, base=base, **kwargs
                )
                return {
                    "status": 201,
                    "message": "Pull request created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "update":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                number = kwargs.pop("number", None)
                if not owner or not repo or not number:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'number' parameter",
                        "data": None,
                    }
                response = client.update_pull_request(
                    owner=owner, repo=repo, number=int(number), **kwargs
                )
                return {
                    "status": 200,
                    "message": "Pull request updated successfully",
                    "data": response.data.model_dump(),
                }
            else:
                return {
                    "status": 400,
                    "error": f"Unknown action: {action}",
                    "data": None,
                }
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}


def register_content_tools(mcp: FastMCP):
    @mcp.tool(tags={"contents"})
    async def github_contents(
        action: str = Field(
            description="Action to perform. Must be one of: 'get', 'create', 'update', 'delete'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub contents."""
        if ctx:
            await ctx.info("Executing github_contents action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "get":
                response = client.get_contents(**kwargs)
                if isinstance(response.data, list):
                    data = [item.model_dump() for item in response.data]
                else:
                    data = response.data.model_dump()
                return {
                    "status": 200,
                    "message": "Contents retrieved successfully",
                    "data": data,
                }
            elif action == "create":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                path = kwargs.pop("path", None)
                message = kwargs.pop("message", None)
                content = kwargs.pop("content", None)
                if not owner or not repo or not path or not message or not content:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', 'path', 'message', or 'content' parameter",
                        "data": None,
                    }
                response = client.create_content(
                    owner=owner,
                    repo=repo,
                    path=path,
                    message=message,
                    content=content,
                    **kwargs,
                )
                return {
                    "status": 201,
                    "message": "Content created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "update":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                path = kwargs.pop("path", None)
                message = kwargs.pop("message", None)
                content = kwargs.pop("content", None)
                sha = kwargs.pop("sha", None)
                if (
                    not owner
                    or not repo
                    or not path
                    or not message
                    or not content
                    or not sha
                ):
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', 'path', 'message', 'content', or 'sha' parameter",
                        "data": None,
                    }
                response = client.update_content(
                    owner=owner,
                    repo=repo,
                    path=path,
                    message=message,
                    content=content,
                    sha=sha,
                    **kwargs,
                )
                return {
                    "status": 200,
                    "message": "Content updated successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "delete":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                path = kwargs.pop("path", None)
                message = kwargs.pop("message", None)
                sha = kwargs.pop("sha", None)
                if not owner or not repo or not path or not message or not sha:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', 'path', 'message', or 'sha' parameter",
                        "data": None,
                    }
                response = client.delete_content(
                    owner=owner,
                    repo=repo,
                    path=path,
                    message=message,
                    sha=sha,
                    **kwargs,
                )
                return {
                    "status": 200,
                    "message": "Content deleted successfully",
                    "data": response.data,
                }
            else:
                return {
                    "status": 400,
                    "error": f"Unknown action: {action}",
                    "data": None,
                }
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}


def register_branch_tools(mcp: FastMCP):
    @mcp.tool(tags={"branches"})
    async def github_branches(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get', 'create', 'delete'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub branches."""
        if ctx:
            await ctx.info("Executing github_branches action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "list":
                response = client.get_branches(**kwargs)
                return {
                    "status": 200,
                    "message": "Branches retrieved successfully",
                    "data": [branch.model_dump() for branch in response.data],
                }
            elif action == "get":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                branch = kwargs.get("branch")
                if not owner or not repo or not branch:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'branch' parameter",
                        "data": None,
                    }
                response = client.get_branch(owner=owner, repo=repo, branch=branch)
                return {
                    "status": 200,
                    "message": "Branch retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "create":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                branch = kwargs.get("branch")
                ref = kwargs.get("ref")
                if not owner or not repo or not branch or not ref:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', 'branch', or 'ref' parameter",
                        "data": None,
                    }
                response = client.create_branch(
                    owner=owner, repo=repo, branch=branch, ref=ref
                )
                return {
                    "status": 201,
                    "message": "Branch created successfully",
                    "data": response.data,
                }
            elif action == "delete":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                branch = kwargs.get("branch")
                if not owner or not repo or not branch:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'branch' parameter",
                        "data": None,
                    }
                response = client.delete_branch(owner=owner, repo=repo, branch=branch)
                return {
                    "status": 200,
                    "message": "Branch deleted successfully",
                    "data": response.data,
                }
            else:
                return {
                    "status": 400,
                    "error": f"Unknown action: {action}",
                    "data": None,
                }
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}


def register_commit_tools(mcp: FastMCP):
    @mcp.tool(tags={"commits"})
    async def github_commits(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub commits."""
        if ctx:
            await ctx.info("Executing github_commits action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "list":
                response = client.get_commits(**kwargs)
                return {
                    "status": 200,
                    "message": "Commits retrieved successfully",
                    "data": [commit.model_dump() for commit in response.data],
                }
            elif action == "get":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                sha = kwargs.get("sha")
                if not owner or not repo or not sha:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'sha' parameter",
                        "data": None,
                    }
                response = client.get_commit(owner=owner, repo=repo, sha=sha)
                return {
                    "status": 200,
                    "message": "Commit retrieved successfully",
                    "data": response.data.model_dump(),
                }
            else:
                return {
                    "status": 400,
                    "error": f"Unknown action: {action}",
                    "data": None,
                }
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}


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

    BRANCHESTOOL = to_boolean(os.getenv("BRANCHESTOOL", "True"))
    if BRANCHESTOOL:
        register_branch_tools(mcp)

    COMMITSTOOL = to_boolean(os.getenv("COMMITSTOOL", "True"))
    if COMMITSTOOL:
        register_commit_tools(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    registered_tags = []
    tools_dict = (
        mcp._tools
        if hasattr(mcp, "_tools")
        else mcp.get_tools()
        if hasattr(mcp, "get_tools")
        else {}
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
