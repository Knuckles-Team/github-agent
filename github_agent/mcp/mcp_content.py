"""MCP tools for content operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import resolve_action
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client

#: Valid content actions for the shared ``resolve_action`` discovery helper.
CONTENT_ACTIONS = ("get", "create", "update", "delete")


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
                response = client.get_contents(**kwargs)
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
                response = client.create_content(
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
                response = client.update_content(
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
                response = client.delete_content(
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
