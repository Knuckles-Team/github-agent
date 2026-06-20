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
from typing import Any

from agent_utilities.mcp_utilities import (
    create_mcp_server,
    load_config,
    register_tool_surface,
    resolve_action,
    run_blocking,
)

from github_agent.api.api_client_orgs import OrganizationCreationNotSupportedError
from github_agent.api_client import Api
from github_agent.auth import allow_destructive_default, get_client
from github_agent.github_response_models import PagesAlreadyEnabled, PagesNotEnabled

__version__ = "1.0.0"
logger = get_logger("GithubMCPServer")
logger.setLevel(logging.INFO)

#: Org actions gated behind allow_destructive / GITHUB_ALLOW_DESTRUCTIVE.
DESTRUCTIVE_ORG_ACTIONS = {"delete", "remove_member"}

#: Repo actions gated behind allow_destructive / GITHUB_ALLOW_DESTRUCTIVE.
DESTRUCTIVE_REPO_ACTIONS = {"pages_delete"}

#: Exact keys dropped by _slim (pure hypermedia/noise, never semantic data).
_SLIM_DROP_EXACT = {"_links", "url", "node_id"}

#: Valid actions per tool — the single source the action Field text and the
#: shared ``resolve_action`` discovery/did-you-mean helper both read from.
REPO_ACTIONS = (
    "list",
    "get",
    "create",
    "delete",
    "update",
    "pages_get",
    "pages_create",
    "pages_update",
    "pages_delete",
    "pages_builds",
    "pages_request_build",
)
ISSUE_ACTIONS = ("list", "get", "create", "update")
PULL_ACTIONS = ("list", "get", "create", "update")
CONTENT_ACTIONS = ("get", "create", "update", "delete")
BRANCH_ACTIONS = (
    "list",
    "get",
    "create",
    "delete",
    "get_protection",
    "update_protection",
    "delete_protection",
)
COMMIT_ACTIONS = ("list", "get")
SEARCH_ACTIONS = ("repositories", "issues", "code")
ORG_ACTIONS = (
    "get",
    "list",
    "update",
    "delete",
    "create",
    "create_repository",
    "repos",
    "members",
    "get_membership",
    "set_membership",
    "remove_member",
    "teams",
)
COLLABORATOR_ACTIONS = ("list", "add", "remove")
WORKFLOW_ACTIONS = (
    "list_workflows",
    "list_runs",
    "get_run",
    "trigger_dispatch",
    "rerun",
    "cancel",
    "delete_run",
)
RELEASE_ACTIONS = ("list", "get", "create", "update", "delete")


def _slim(obj: Any) -> Any:
    """Recursively drop hypermedia ``*_url`` hrefs and ``_links`` noise.

    GitHub list items carry ~15 ``*_url`` API hrefs (plus more nested under
    actor/repository), which dominate the byte size while carrying no data a
    caller acts on. This keeps every real field — including ``html_url`` — and
    strips only the noise, so large list responses stay small. Applied to list
    endpoints when ``slim`` is true (the default); pass ``slim=false`` for the
    full objects.
    """
    if isinstance(obj, list):
        return [_slim(item) for item in obj]
    if isinstance(obj, dict):
        return {
            k: _slim(v)
            for k, v in obj.items()
            if k not in _SLIM_DROP_EXACT
            and not (k.endswith("_url") and k != "html_url")
        }
    return obj


def register_repo_tools(mcp: FastMCP):
    @mcp.tool(tags={"repos"})
    async def github_repos(
        action: str = Field(
            description=(
                "Action to perform. Must be one of: 'list', 'get', 'create', "
                "'delete', 'update', 'pages_get', 'pages_create', "
                "'pages_update', 'pages_delete', 'pages_builds', "
                "'pages_request_build'"
            )
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        allow_destructive: bool = Field(
            default=False,
            description=(
                "Must be true to run destructive actions: ['pages_delete']. "
                "Deleting a Pages site takes it offline immediately."
            ),
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub repositories and their GitHub Pages sites.

        Repository actions (parameters via params_json):
        - 'list': {"visibility", "affiliation", "type"} — repositories for
          the authenticated user.
        - 'get': {"owner", "repo"} — a single repository.
        - 'create': {"name", plus payload fields such as description,
          private, auto_init, ...} — create a repository for the
          authenticated user.
        - 'delete': {"owner", "repo"} — delete a repository.
        - 'update': {"owner", "repo", plus mutable repository settings} —
          PATCH /repos/{owner}/{repo}.

        GitHub Pages actions:
        - 'pages_get': {"owner", "repo"} — the Pages site configuration;
          returns status 404 with a typed not-enabled result when Pages is
          off.
        - 'pages_create': {"owner", "repo", "build_type":
          "workflow"|"legacy", "source": {"branch", "path"}} — enable
          Pages. build_type defaults to 'workflow'; 'legacy' requires
          source. Returns status 409 with a typed already-enabled result
          when Pages is already on.
        - 'pages_update': {"owner", "repo", plus build_type, source, cname,
          https_enforced} — change the Pages configuration.
        - 'pages_delete': {"owner", "repo"} — disable Pages and delete the
          site; requires allow_destructive=true.
        - 'pages_builds': {"owner", "repo", "latest": true|false} — list
          Pages builds, or only the latest build with latest=true.
        - 'pages_request_build': {"owner", "repo"} — request a fresh Pages
          build without pushing a commit (the programmatic fix for the
          first-deploy race where the initial Pages build never ran).
        """
        if ctx:
            await ctx.info("Executing github_repos action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        slim = kwargs.pop("slim", True)

        resolved = resolve_action(action, REPO_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action in DESTRUCTIVE_REPO_ACTIONS and not (
            allow_destructive is True or allow_destructive_default()
        ):
            return {
                "status": 403,
                "error": (
                    f"Action '{action}' is destructive and blocked by default. "
                    "Re-run with allow_destructive=true (or set "
                    "GITHUB_ALLOW_DESTRUCTIVE=True) to confirm."
                ),
                "data": None,
            }

        try:
            if action == "list":
                response = await run_blocking(client.get_repositories, **kwargs)
                data = [repo.model_dump() for repo in response.data]
                return {
                    "status": 200,
                    "message": "Repositories retrieved successfully",
                    "data": _slim(data) if slim else data,
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
                response = await run_blocking(
                    client.get_repository, owner=owner, repo=repo
                )
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
                response = await run_blocking(
                    client.create_repository, name=name, **kwargs
                )
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
                response = await run_blocking(
                    client.delete_repository, owner=owner, repo=repo
                )
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
                response = await run_blocking(
                    client.update_repository, owner=owner, repo=repo, **kwargs
                )
                return {
                    "status": 200,
                    "message": "Repository updated successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "pages_get":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(client.get_pages, owner=owner, repo=repo)
                if isinstance(response.data, PagesNotEnabled):
                    return {
                        "status": 404,
                        "message": "GitHub Pages is not enabled for this repository",
                        "data": response.data.model_dump(),
                    }
                return {
                    "status": 200,
                    "message": "Pages site retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "pages_create":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.create_pages,
                    owner=owner,
                    repo=repo,
                    build_type=kwargs.get("build_type", "workflow"),
                    source=kwargs.get("source"),
                )
                if isinstance(response.data, PagesAlreadyEnabled):
                    return {
                        "status": 409,
                        "message": "GitHub Pages is already enabled for this repository",
                        "data": response.data.model_dump(),
                    }
                return {
                    "status": 201,
                    "message": "Pages site created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "pages_update":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.update_pages, owner=owner, repo=repo, **kwargs
                )
                return {
                    "status": 200,
                    "message": "Pages site updated successfully",
                    "data": response.data,
                }
            elif action == "pages_delete":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.delete_pages, owner=owner, repo=repo
                )
                return {
                    "status": 200,
                    "message": "Pages site deleted successfully",
                    "data": response.data,
                }
            elif action == "pages_builds":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                if kwargs.pop("latest", False):
                    response = await run_blocking(
                        client.get_pages_build_latest, owner=owner, repo=repo
                    )
                    return {
                        "status": 200,
                        "message": "Latest Pages build retrieved successfully",
                        "data": response.data.model_dump(),
                    }
                response = await run_blocking(
                    client.list_pages_builds, owner=owner, repo=repo, **kwargs
                )
                return {
                    "status": 200,
                    "message": "Pages builds retrieved successfully",
                    "data": [build.model_dump() for build in response.data],
                }
            elif action == "pages_request_build":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.request_pages_build, owner=owner, repo=repo
                )
                return {
                    "status": 201,
                    "message": "Pages build requested successfully",
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
        """Manage GitHub issues.

        list params (via params_json): owner, repo, and optional filters applied
        server-side — state (open/closed/all), labels, assignee, since, per_page
        (1-100, default 30), max_pages (default 1 page; max_pages<=0 = all pages).
        Note: the list/search APIs return PRs alongside issues — a returned item
        with a 'pull_request' field is a PR, not an issue.
        """
        if ctx:
            await ctx.info("Executing github_issues action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, ISSUE_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "list":
                response = await run_blocking(client.get_issues, **kwargs)
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
                response = await run_blocking(
                    client.get_issue, owner=owner, repo=repo, number=int(number)
                )
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
                response = await run_blocking(
                    client.create_issue, owner=owner, repo=repo, title=title, **kwargs
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
                response = await run_blocking(
                    client.update_issue,
                    owner=owner,
                    repo=repo,
                    number=int(number),
                    **kwargs,
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
        """Manage GitHub pull requests.

        list params (via params_json): owner, repo, and optional filters applied
        server-side — state (open/closed/all), head, base, sort, direction,
        per_page (1-100, default 30), max_pages (default 1 page; max_pages<=0 =
        all pages).
        """
        if ctx:
            await ctx.info("Executing github_pulls action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, PULL_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "list":
                response = await run_blocking(client.get_pull_requests, **kwargs)
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
                response = await run_blocking(
                    client.get_pull_request, owner=owner, repo=repo, number=int(number)
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
                response = await run_blocking(
                    client.create_pull_request,
                    owner=owner,
                    repo=repo,
                    title=title,
                    head=head,
                    base=base,
                    **kwargs,
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
                response = await run_blocking(
                    client.update_pull_request,
                    owner=owner,
                    repo=repo,
                    number=int(number),
                    **kwargs,
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

        resolved = resolve_action(action, CONTENT_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "get":
                response = await run_blocking(client.get_contents, **kwargs)
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
                response = await run_blocking(
                    client.create_content,
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
                response = await run_blocking(
                    client.update_content,
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
                response = await run_blocking(
                    client.delete_content,
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
            description="Action to perform. Must be one of: 'list', 'get', 'create', 'delete', 'get_protection', 'update_protection', 'delete_protection'"
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

        resolved = resolve_action(action, BRANCH_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "list":
                response = await run_blocking(client.get_branches, **kwargs)
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
                response = await run_blocking(
                    client.get_branch, owner=owner, repo=repo, branch=branch
                )
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
                response = await run_blocking(
                    client.create_branch, owner=owner, repo=repo, branch=branch, ref=ref
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
                response = await run_blocking(
                    client.delete_branch, owner=owner, repo=repo, branch=branch
                )
                return {
                    "status": 200,
                    "message": "Branch deleted successfully",
                    "data": response.data,
                }
            elif action == "get_protection":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                branch = kwargs.get("branch")
                if not owner or not repo or not branch:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'branch' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_branch_protection, owner=owner, repo=repo, branch=branch
                )
                return {
                    "status": 200,
                    "message": "Branch protection retrieved successfully",
                    "data": response.data,
                }
            elif action == "update_protection":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                branch = kwargs.get("branch")
                protection_config = kwargs.get("protection_config")
                if not owner or not repo or not branch or protection_config is None:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', 'branch', or 'protection_config' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.update_branch_protection,
                    owner=owner,
                    repo=repo,
                    branch=branch,
                    protection_config=protection_config,
                )
                return {
                    "status": 200,
                    "message": "Branch protection updated successfully",
                    "data": response.data,
                }
            elif action == "delete_protection":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                branch = kwargs.get("branch")
                if not owner or not repo or not branch:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'branch' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.delete_branch_protection,
                    owner=owner,
                    repo=repo,
                    branch=branch,
                )
                return {
                    "status": 200,
                    "message": "Branch protection deleted successfully",
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

        resolved = resolve_action(action, COMMIT_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "list":
                response = await run_blocking(client.get_commits, **kwargs)
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
                response = await run_blocking(
                    client.get_commit, owner=owner, repo=repo, sha=sha
                )
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


def register_search_tools(mcp: FastMCP):
    @mcp.tool(tags={"search"})
    async def github_search(
        action: str = Field(
            description="Action to perform. Must be one of: 'repositories', 'issues', 'code'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Search GitHub repositories, issues, or code."""
        if ctx:
            await ctx.info("Executing github_search action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, SEARCH_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "repositories":
                response = await run_blocking(client.search_repositories, **kwargs)
                return {
                    "status": 200,
                    "message": "Repositories searched successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "issues":
                response = await run_blocking(client.search_issues, **kwargs)
                return {
                    "status": 200,
                    "message": "Issues searched successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "code":
                response = await run_blocking(client.search_code, **kwargs)
                return {
                    "status": 200,
                    "message": "Code searched successfully",
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


def register_org_tools(mcp: FastMCP):
    @mcp.tool(tags={"orgs"})
    async def github_orgs(
        action: str = Field(
            description=(
                "Action to perform. Must be one of: 'get', 'list', 'update', "
                "'delete', 'create', 'create_repository', 'repos', 'members', "
                "'get_membership', 'set_membership', 'remove_member', 'teams'"
            )
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        allow_destructive: bool = Field(
            default=False,
            description=(
                "Must be true to run destructive actions: ['delete', "
                "'remove_member']. Organization deletion is IRREVERSIBLE."
            ),
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub organizations.

        Actions (parameters via params_json):
        - 'get': {"org"} — full organization profile (GET /orgs/{org}).
        - 'list': {"scope": "member"|"all", "since"} — scope 'member'
          (default) lists the authenticated user's organizations
          (GET /user/orgs); scope 'all' lists every organization
          (GET /organizations, paginated by the 'since' ID cursor).
        - 'update': {"org", plus mutable fields: billing_email, company,
          email, location, name, description, blog, twitter_username,
          has_organization_projects, has_repository_projects,
          default_repository_permission, members_can_create_repositories,
          web_commit_signoff_required, ...} — PATCH /orgs/{org}.
        - 'delete': {"org"} — IRREVERSIBLE; schedules organization deletion
          (HTTP 202). Requires allow_destructive=true.
        - 'create': {"login", "admin", "profile_name"} — GitHub Enterprise
          Server ONLY (POST /admin/organizations). github.com organizations
          CANNOT be created via the API (web UI only); against
          api.github.com this action returns an error explaining that.
        - 'create_repository': {"org", "name", plus the same payload fields
          as the github_repos 'create' action} — POST /orgs/{org}/repos.
        - 'repos': {"org", "type"} — list organization repositories.
        - 'members': {"org", "role"} — list organization members.
        - 'get_membership': {"org", "username"} — a user's membership state
          and role.
        - 'set_membership': {"org", "username", "role"} — add or update a
          member ('member' or 'admin'); invites the user if not yet a member.
        - 'remove_member': {"org", "username"} — remove a member; requires
          allow_destructive=true.
        - 'teams': {"org"} — list organization teams.
        """
        if ctx:
            await ctx.info("Executing github_orgs action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, ORG_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action in DESTRUCTIVE_ORG_ACTIONS and not (
            allow_destructive is True or allow_destructive_default()
        ):
            return {
                "status": 403,
                "error": (
                    f"Action '{action}' is destructive and blocked by default. "
                    "Re-run with allow_destructive=true (or set "
                    "GITHUB_ALLOW_DESTRUCTIVE=True) to confirm."
                ),
                "data": None,
            }

        try:
            if action == "get":
                org = kwargs.get("org")
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = await run_blocking(client.get_organization, org=org)
                return {
                    "status": 200,
                    "message": "Organization retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "list":
                response = await run_blocking(client.list_organizations, **kwargs)
                return {
                    "status": 200,
                    "message": "Organizations retrieved successfully",
                    "data": [org.model_dump() for org in response.data],
                }
            elif action == "update":
                org = kwargs.pop("org", None)
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.update_organization, org=org, **kwargs
                )
                return {
                    "status": 200,
                    "message": "Organization updated successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "delete":
                org = kwargs.get("org")
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = await run_blocking(client.delete_organization, org=org)
                return {
                    "status": 202,
                    "message": "Organization deletion scheduled (irreversible)",
                    "data": response.data,
                }
            elif action == "create":
                login = kwargs.get("login")
                admin = kwargs.get("admin")
                if not login or not admin:
                    return {
                        "status": 400,
                        "error": "Missing 'login' or 'admin' parameter",
                        "data": None,
                    }
                try:
                    response = await run_blocking(
                        client.create_organization,
                        login=login,
                        admin=admin,
                        profile_name=kwargs.get("profile_name"),
                    )
                except OrganizationCreationNotSupportedError as e:
                    return {"status": 400, "error": str(e), "data": None}
                return {
                    "status": 201,
                    "message": "Organization created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "create_repository":
                org = kwargs.pop("org", None)
                name = kwargs.pop("name", None)
                if not org or not name:
                    return {
                        "status": 400,
                        "error": "Missing 'org' or 'name' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.create_organization_repository, org=org, name=name, **kwargs
                )
                return {
                    "status": 201,
                    "message": "Organization repository created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "repos":
                response = await run_blocking(client.get_org_repos, **kwargs)
                return {
                    "status": 200,
                    "message": "Organization repositories retrieved successfully",
                    "data": [repo.model_dump() for repo in response.data],
                }
            elif action == "members":
                response = await run_blocking(client.get_org_members, **kwargs)
                return {
                    "status": 200,
                    "message": "Organization members retrieved successfully",
                    "data": [member.model_dump() for member in response.data],
                }
            elif action == "get_membership":
                org = kwargs.get("org")
                username = kwargs.get("username")
                if not org or not username:
                    return {
                        "status": 400,
                        "error": "Missing 'org' or 'username' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_organization_membership, org=org, username=username
                )
                return {
                    "status": 200,
                    "message": "Organization membership retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "set_membership":
                org = kwargs.get("org")
                username = kwargs.get("username")
                if not org or not username:
                    return {
                        "status": 400,
                        "error": "Missing 'org' or 'username' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.set_organization_membership,
                    org=org,
                    username=username,
                    role=kwargs.get("role", "member"),
                )
                return {
                    "status": 200,
                    "message": "Organization membership set successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "remove_member":
                org = kwargs.get("org")
                username = kwargs.get("username")
                if not org or not username:
                    return {
                        "status": 400,
                        "error": "Missing 'org' or 'username' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.remove_organization_member, org=org, username=username
                )
                return {
                    "status": 200,
                    "message": "Organization member removed successfully",
                    "data": response.data,
                }
            elif action == "teams":
                org = kwargs.get("org")
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = await run_blocking(client.get_org_teams, org=org)
                return {
                    "status": 200,
                    "message": "Organization teams retrieved successfully",
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


def register_collaborator_tools(mcp: FastMCP):
    @mcp.tool(tags={"collaborators"})
    async def github_collaborators(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'add', 'remove'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage repository collaborators."""
        if ctx:
            await ctx.info("Executing github_collaborators action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, COLLABORATOR_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "list":
                response = await run_blocking(client.get_collaborators, **kwargs)
                return {
                    "status": 200,
                    "message": "Collaborators retrieved successfully",
                    "data": [c.model_dump() for c in response.data],
                }
            elif action == "add":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                username = kwargs.get("username")
                permission = kwargs.get("permission")
                if not owner or not repo or not username:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'username' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.add_collaborator,
                    owner=owner,
                    repo=repo,
                    username=username,
                    permission=permission,
                )
                return {
                    "status": 200,
                    "message": "Collaborator added successfully",
                    "data": response.data,
                }
            elif action == "remove":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                username = kwargs.get("username")
                if not owner or not repo or not username:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'username' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.remove_collaborator,
                    owner=owner,
                    repo=repo,
                    username=username,
                )
                return {
                    "status": 200,
                    "message": "Collaborator removed successfully",
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


def register_action_tools(mcp: FastMCP):
    @mcp.tool(tags={"actions"})
    async def github_actions(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_workflows', 'list_runs', 'get_run', 'trigger_dispatch', 'rerun', 'cancel', 'delete_run'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub actions workflows and runs.

        list_runs params (via params_json): owner, repo, and optional filters
        applied server-side to keep results small — status (e.g. failure,
        success, completed, in_progress), branch, per_page (1-100, default 30),
        max_pages (default 1 page; max_pages<=0 = all pages), slim (default
        true: drops hypermedia *_url/_links noise, keeps html_url and all data).
        Prefer status=failure + branch=<default> for a CI-health sweep.
        """
        if ctx:
            await ctx.info("Executing github_actions action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        slim = kwargs.pop("slim", True)

        resolved = resolve_action(action, WORKFLOW_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "list_workflows":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing required 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_workflows, owner=owner, repo=repo
                )
                return {
                    "status": 200,
                    "message": "Workflows retrieved successfully",
                    "data": [w.model_dump() for w in response.data],
                }
            elif action == "list_runs":
                response = await run_blocking(client.get_workflow_runs, **kwargs)
                data = [r.model_dump() for r in response.data]
                return {
                    "status": 200,
                    "message": "Workflow runs retrieved successfully",
                    "data": _slim(data) if slim else data,
                }
            elif action == "get_run":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                run_id = kwargs.get("run_id")
                if not owner or not repo or not run_id:
                    return {
                        "status": 400,
                        "error": "Missing required 'owner', 'repo', or 'run_id' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_workflow_run, owner=owner, repo=repo, run_id=int(run_id)
                )
                return {
                    "status": 200,
                    "message": "Workflow run retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "trigger_dispatch":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                workflow_id = kwargs.get("workflow_id")
                ref = kwargs.get("ref")
                inputs = kwargs.get("inputs")
                if not owner or not repo or not workflow_id or not ref:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', 'workflow_id', or 'ref' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.trigger_workflow_dispatch,
                    owner=owner,
                    repo=repo,
                    workflow_id=workflow_id,
                    ref=ref,
                    inputs=inputs,
                )
                return {
                    "status": 200,
                    "message": "Workflow dispatch triggered successfully",
                    "data": response.data,
                }
            elif action == "rerun":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                run_id = kwargs.get("run_id")
                if not owner or not repo or not run_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'run_id' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.rerun_workflow_run,
                    owner=owner,
                    repo=repo,
                    run_id=int(run_id),
                )
                return {
                    "status": 200,
                    "message": "Workflow run rerun triggered",
                    "data": response.data,
                }
            elif action == "cancel":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                run_id = kwargs.get("run_id")
                if not owner or not repo or not run_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'run_id' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.cancel_workflow_run,
                    owner=owner,
                    repo=repo,
                    run_id=int(run_id),
                )
                return {
                    "status": 200,
                    "message": "Workflow run cancellation triggered",
                    "data": response.data,
                }
            elif action == "delete_run":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                run_id = kwargs.get("run_id")
                if not owner or not repo or not run_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'run_id' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.delete_workflow_run,
                    owner=owner,
                    repo=repo,
                    run_id=int(run_id),
                )
                return {
                    "status": 200,
                    "message": "Workflow run deleted successfully",
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


def register_release_tools(mcp: FastMCP):
    @mcp.tool(tags={"releases"})
    async def github_releases(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get', 'create', 'update', 'delete'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage repository releases."""
        if ctx:
            await ctx.info("Executing github_releases action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, RELEASE_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "list":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_releases, owner=owner, repo=repo
                )
                return {
                    "status": 200,
                    "message": "Releases retrieved successfully",
                    "data": [r.model_dump() for r in response.data],
                }
            elif action == "get":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                release_id = kwargs.get("release_id")
                if not owner or not repo or not release_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'release_id' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_release,
                    owner=owner,
                    repo=repo,
                    release_id=int(release_id),
                )
                return {
                    "status": 200,
                    "message": "Release retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "create":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                tag_name = kwargs.pop("tag_name", None)
                if not owner or not repo or not tag_name:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'tag_name' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.create_release,
                    owner=owner,
                    repo=repo,
                    tag_name=tag_name,
                    **kwargs,
                )
                return {
                    "status": 201,
                    "message": "Release created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "update":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                release_id = kwargs.pop("release_id", None)
                if not owner or not repo or not release_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'release_id' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.update_release,
                    owner=owner,
                    repo=repo,
                    release_id=int(release_id),
                    **kwargs,
                )
                return {
                    "status": 200,
                    "message": "Release updated successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "delete":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                release_id = kwargs.get("release_id")
                if not owner or not repo or not release_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'release_id' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.delete_release,
                    owner=owner,
                    repo=repo,
                    release_id=int(release_id),
                )
                return {
                    "status": 200,
                    "message": "Release deleted successfully",
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


#: (tag, env-toggle, registrar) — explicit so the historical env-var names
#: (REPOSTOOL/PULLSTOOL/CONTENTSTOOL/… with trailing "S") are preserved rather
#: than auto-derived from the function names.
TOOL_REGISTRY = [
    ("repos", "REPOSTOOL", register_repo_tools),
    ("issue", "ISSUETOOL", register_issue_tools),
    ("pulls", "PULLSTOOL", register_pull_tools),
    ("contents", "CONTENTSTOOL", register_content_tools),
    ("branches", "BRANCHESTOOL", register_branch_tools),
    ("commits", "COMMITSTOOL", register_commit_tools),
    ("search", "SEARCHTOOL", register_search_tools),
    ("orgs", "ORGSTOOL", register_org_tools),
    ("collaborators", "COLLABORATORSTOOL", register_collaborator_tools),
    ("actions", "ACTIONSTOOL", register_action_tools),
    ("releases", "RELEASESTOOL", register_release_tools),
]


def get_mcp_instance() -> tuple[Any, Any, Any, Any, Any]:
    load_config()
    args, mcp, middlewares = create_mcp_server(
        name="Github MCP",
        version=__version__,
        instructions="Github MCP Server - Manage your repositories, issues, and pull requests.",
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
