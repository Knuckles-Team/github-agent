"""MCP tools for branch operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import resolve_action
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client

#: Valid branch actions for the shared ``resolve_action`` discovery helper.
BRANCH_ACTIONS = (
    "list",
    "get",
    "create",
    "delete",
    "get_protection",
    "update_protection",
    "delete_protection",
)


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

        resolved = resolve_action(action, BRANCH_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

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
