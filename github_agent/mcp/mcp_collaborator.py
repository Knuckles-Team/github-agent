"""MCP tools for collaborator operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client

#: Valid collaborator actions for the shared ``resolve_action`` discovery helper.
COLLABORATOR_ACTIONS = ("list", "add", "remove")


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
            return {
                "status": 400,
                "error": f"Invalid params_json: {type(e).__name__}",
                "data": None,
            }

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
