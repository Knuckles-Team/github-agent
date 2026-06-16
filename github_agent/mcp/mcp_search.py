"""MCP tools for search operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import resolve_action, run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client

#: Valid search actions for the shared ``resolve_action`` discovery helper.
SEARCH_ACTIONS = ("repositories", "issues", "code")


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
