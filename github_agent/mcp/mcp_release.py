"""MCP tools for release operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import resolve_action
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client

#: Valid release actions for the shared ``resolve_action`` discovery helper.
RELEASE_ACTIONS = ("list", "get", "create", "update", "delete")


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
                response = client.get_releases(owner=owner, repo=repo)
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
                response = client.get_release(
                    owner=owner, repo=repo, release_id=int(release_id)
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
                response = client.create_release(
                    owner=owner, repo=repo, tag_name=tag_name, **kwargs
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
                response = client.update_release(
                    owner=owner, repo=repo, release_id=int(release_id), **kwargs
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
                response = client.delete_release(
                    owner=owner, repo=repo, release_id=int(release_id)
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
