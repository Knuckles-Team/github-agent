"""MCP tools for org operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client


def register_org_tools(mcp: FastMCP):
    @mcp.tool(tags={"orgs"})
    async def github_orgs(
        action: str = Field(
            description="Action to perform. Must be one of: 'repos', 'members', 'teams'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub organizations."""
        if ctx:
            await ctx.info("Executing github_orgs action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "repos":
                response = client.get_org_repos(**kwargs)
                return {
                    "status": 200,
                    "message": "Organization repositories retrieved successfully",
                    "data": [repo.model_dump() for repo in response.data],
                }
            elif action == "members":
                response = client.get_org_members(**kwargs)
                return {
                    "status": 200,
                    "message": "Organization members retrieved successfully",
                    "data": [member.model_dump() for member in response.data],
                }
            elif action == "teams":
                org = kwargs.get("org")
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = client.get_org_teams(org=org)
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
