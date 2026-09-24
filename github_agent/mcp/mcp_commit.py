"""MCP tools for commit operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_connector_sdk.mcp.action_dispatch import resolve_action
from agent_connector_sdk.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client

#: Valid commit actions for the shared ``resolve_action`` discovery helper.
COMMIT_ACTIONS = ("list", "get")


async def _list_commits(client, kwargs: dict) -> dict:
    response = await run_blocking(client.get_commits, **kwargs)
    return {
        "status": 200,
        "message": "Commits retrieved successfully",
        "data": [commit.model_dump() for commit in response.data],
    }


async def _get_commit(client, kwargs: dict) -> dict:
    owner = kwargs.get("owner")
    repo = kwargs.get("repo")
    sha = kwargs.get("sha")
    if not owner or not repo or not sha:
        return {
            "status": 400,
            "error": "Missing 'owner', 'repo', or 'sha' parameter",
            "data": None,
        }
    response = await run_blocking(client.get_commit, owner=owner, repo=repo, sha=sha)
    return {
        "status": 200,
        "message": "Commit retrieved successfully",
        "data": response.data.model_dump(),
    }


#: Dispatch table for the resolved commit action -> its async handler.
_COMMIT_ACTION_HANDLERS = {
    "list": _list_commits,
    "get": _get_commit,
}


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
            return {
                "status": 400,
                "error": f"Invalid params_json: {type(e).__name__}",
                "data": None,
            }

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, COMMIT_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        handler = _COMMIT_ACTION_HANDLERS.get(action)
        if handler is None:
            return {"status": 400, "error": f"Unknown action: {action}", "data": None}

        try:
            return await handler(client, kwargs)
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}
