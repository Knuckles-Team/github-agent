"""MCP tools for issue operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client


def register_issue_tools(mcp: FastMCP):
    @mcp.tool(tags={"issues"})
    async def github_issues(
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
        """Manage GitHub issues.

        list params (via params_json): owner, repo, and optional filters applied
        server-side — state (open/closed/all), labels, assignee, since, per_page
        (1-100, default 30), max_pages (default 1 page; max_pages<=0 = all pages).
        Note: the list/search APIs return PRs alongside issues — a returned item
        with a 'pull_request' field is a PR, not an issue.
        """
        if ctx:
            await ctx.info("Executing github_issues action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "list":
                response = client.get_issues(**kwargs)
                return {
                    "status": 200,
                    "message": "Issues retrieved successfully",
                    "data": [issue.model_dump() for issue in response.data],
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
                response = client.get_issue(owner=owner, repo=repo, number=int(number))
                return {
                    "status": 200,
                    "message": "Issue retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "create":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                title = kwargs.pop("title", None)
                if not owner or not repo or not title:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'title' parameter",
                        "data": None,
                    }
                response = client.create_issue(
                    owner=owner, repo=repo, title=title, **kwargs
                )
                return {
                    "status": 201,
                    "message": "Issue created successfully",
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
                response = client.update_issue(
                    owner=owner, repo=repo, number=int(number), **kwargs
                )
                return {
                    "status": 200,
                    "message": "Issue updated successfully",
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
