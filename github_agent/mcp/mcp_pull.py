"""MCP tools for pull operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client


def register_pull_tools(mcp: FastMCP):
    @mcp.tool(tags={"pulls"})
    async def github_pulls(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get', 'create', 'update'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub pull requests.

        list params (via params_json): owner, repo, and optional filters applied
        server-side — state (open/closed/all), head, base, sort, direction,
        per_page (1-100, default 30), max_pages (default 1 page; max_pages<=0 =
        all pages).
        """
        if ctx:
            await ctx.info("Executing github_pulls action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "list":
                response = client.get_pull_requests(**kwargs)
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
                response = client.get_pull_request(
                    owner=owner, repo=repo, number=int(number)
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
                response = client.create_pull_request(
                    owner=owner, repo=repo, title=title, head=head, base=base, **kwargs
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
                response = client.update_pull_request(
                    owner=owner, repo=repo, number=int(number), **kwargs
                )
                return {
                    "status": 200,
                    "message": "Pull request updated successfully",
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
