"""MCP tools for repo operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client


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
