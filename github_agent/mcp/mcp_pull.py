"""MCP tools for pull operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import (
    allow_destructive_default,
    get_client,
    get_graphql_client,
)

#: Valid pull-request actions for the shared ``resolve_action`` discovery helper.
PULL_ACTIONS = (
    "list",
    "get",
    "create",
    "update",
    "approve",
    "request_reviewers",
    "merge",
    "enable_auto_merge",
    "disable_auto_merge",
)

#: Pull actions gated behind allow_destructive / GITHUB_ALLOW_DESTRUCTIVE.
DESTRUCTIVE_PULL_ACTIONS = {"merge", "enable_auto_merge"}


def register_pull_tools(mcp: FastMCP):
    @mcp.tool(tags={"pulls"})
    async def github_pulls(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get', 'create', 'update', 'approve', 'request_reviewers', 'merge', 'enable_auto_merge', 'disable_auto_merge'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        allow_destructive: bool = Field(
            default=False,
            description="Confirm a guarded write ('merge', 'enable_auto_merge'). Also honoured via GITHUB_ALLOW_DESTRUCTIVE.",
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub pull requests and their review/merge lifecycle.

        list params (via params_json): owner, repo, and optional filters applied
        server-side — state (open/closed/all), head, base, sort, direction,
        per_page (1-100, default 30), max_pages (default 1 page; max_pages<=0 =
        all pages). Write actions: approve, request_reviewers, merge (merge_method
        merge/squash/rebase), enable_auto_merge/disable_auto_merge (GraphQL;
        accept owner+repo+number or a pull_request_id node id). merge and
        enable_auto_merge are guarded by allow_destructive /
        GITHUB_ALLOW_DESTRUCTIVE.
        """
        if ctx:
            await ctx.info("Executing github_pulls action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {
                "status": 400,
                "error": f"Invalid params_json: {type(e).__name__}",
                "data": None,
            }

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, PULL_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action in DESTRUCTIVE_PULL_ACTIONS and not (
            allow_destructive is True or allow_destructive_default()
        ):
            return {
                "status": 403,
                "error": (
                    f"Action '{action}' is a guarded write and blocked by default. "
                    "Re-run with allow_destructive=true (or set "
                    "GITHUB_ALLOW_DESTRUCTIVE=True) to confirm."
                ),
                "data": None,
            }

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
            elif action == "approve":
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
                    client.create_pull_request_review,
                    owner=owner,
                    repo=repo,
                    number=int(number),
                    event=kwargs.get("event", "APPROVE"),
                    body=kwargs.get("body"),
                )
                return {
                    "status": 200,
                    "message": "Pull request review submitted successfully",
                    "data": response.data,
                }
            elif action == "request_reviewers":
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
                    client.request_reviewers,
                    owner=owner,
                    repo=repo,
                    number=int(number),
                    reviewers=kwargs.get("reviewers"),
                    team_reviewers=kwargs.get("team_reviewers"),
                )
                return {
                    "status": 200,
                    "message": "Reviewers requested successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "merge":
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
                    client.merge_pull_request,
                    owner=owner,
                    repo=repo,
                    number=int(number),
                    merge_method=kwargs.get("merge_method", "merge"),
                    commit_title=kwargs.get("commit_title"),
                    commit_message=kwargs.get("commit_message"),
                    sha=kwargs.get("sha"),
                )
                return {
                    "status": 200,
                    "message": "Pull request merged successfully",
                    "data": response.data,
                }
            elif action in ("enable_auto_merge", "disable_auto_merge"):
                # GraphQL-only actions: resolve the gql client lazily, here
                # and only here, so a construction failure (e.g. no token,
                # unreachable endpoint) can never break the REST actions
                # above (list/get/create/update/approve/request_reviewers/
                # merge), which don't need a GraphQL client at all.
                try:
                    gql_client = await run_blocking(get_graphql_client)
                except Exception as e:
                    return {
                        "status": 500,
                        "error": f"GraphQL client unavailable: {type(e).__name__}",
                        "data": None,
                    }
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                number = kwargs.get("number")
                node_id = kwargs.get("pull_request_id")
                if not node_id:
                    if not owner or not repo or not number:
                        return {
                            "status": 400,
                            "error": "Provide 'pull_request_id' (node id) or 'owner'+'repo'+'number'",
                            "data": None,
                        }
                    pr = await run_blocking(
                        client.get_pull_request,
                        owner=owner,
                        repo=repo,
                        number=int(number),
                    )
                    node_id = pr.data.node_id
                if action == "enable_auto_merge":
                    data = await run_blocking(
                        gql_client.enable_pull_request_auto_merge,
                        pull_request_id=node_id,
                        merge_method=kwargs.get("merge_method", "MERGE"),
                    )
                    message = "Auto-merge enabled successfully"
                else:
                    data = await run_blocking(
                        gql_client.disable_pull_request_auto_merge,
                        pull_request_id=node_id,
                    )
                    message = "Auto-merge disabled successfully"
                return {"status": 200, "message": message, "data": data}
            else:
                return {
                    "status": 400,
                    "error": f"Unknown action: {action}",
                    "data": None,
                }
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}
