"""MCP tools for commit operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import resolve_action
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client

#: Valid commit actions for the shared ``resolve_action`` discovery helper.
COMMIT_ACTIONS = ("list", "get")


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
