"""MCP tools for issue operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import resolve_action, run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client

#: Valid issue actions for the shared ``resolve_action`` discovery helper.
ISSUE_ACTIONS = ("list", "get", "create", "update")


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

        list params (via params_json): EITHER repo-scoped (owner + repo) OR
        org-wide (org, no repo). Pass ``org`` alone to fetch issues across EVERY
        repo in the organization in a SINGLE search call (GitHub /search/issues
        with ``org:<org> is:issue``) — far faster than listing repos and paging
        issues per-repo. Optional filters applied server-side: state
        (open/closed/all, default open), labels (comma-separated), assignee,
        since, per_page (1-100, default 30), max_pages (default 1 page;
        max_pages<=0 = all pages). Note: a returned item with a 'pull_request'
        field is a PR, not an issue (the org path excludes PRs via is:issue).
        """
        if ctx:
            await ctx.info("Executing github_issues action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, ISSUE_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        try:
            if action == "list":
                org = kwargs.get("org")
                if org and not kwargs.get("repo"):
                    # Org-wide: ONE /search/issues call (org:<org> is:issue) instead
                    # of enumerate-repos + page-issues-per-repo (N+1 calls). Translate
                    # the list filters into search qualifiers.
                    state = str(kwargs.get("state", "open")).lower()
                    qualifiers = [f"org:{org}", "is:issue"]
                    if state in ("open", "closed"):
                        qualifiers.append(f"state:{state}")
                    if kwargs.get("assignee"):
                        qualifiers.append(f"assignee:{kwargs['assignee']}")
                    for label in str(kwargs.get("labels", "")).split(","):
                        label = label.strip()
                        if label:
                            qualifiers.append(f'label:"{label}"')
                    search_kwargs: dict = {"q": " ".join(qualifiers)}
                    for k in ("sort", "order", "per_page", "max_pages"):
                        if kwargs.get(k) is not None:
                            search_kwargs[k] = kwargs[k]
                    response = await run_blocking(
                        client.search_issues, **search_kwargs
                    )
                    return {
                        "status": 200,
                        "message": f"Org-wide issues for '{org}' via search (1 call)",
                        "data": response.data.items,
                    }
                response = await run_blocking(client.get_issues, **kwargs)
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
                response = await run_blocking(
                    client.get_issue, owner=owner, repo=repo, number=int(number)
                )
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
                response = await run_blocking(
                    client.create_issue, owner=owner, repo=repo, title=title, **kwargs
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
                response = await run_blocking(
                    client.update_issue,
                    owner=owner,
                    repo=repo,
                    number=int(number),
                    **kwargs,
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
