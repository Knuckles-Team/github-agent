"""MCP tools for Dependabot alert operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import allow_destructive_default, get_client

#: Dependabot actions gated behind allow_destructive / GITHUB_ALLOW_DESTRUCTIVE.
DESTRUCTIVE_DEPENDABOT_ACTIONS = {"update"}

#: Valid Dependabot actions for the shared ``resolve_action`` discovery helper.
DEPENDABOT_ACTIONS = ("list", "get", "list_org", "update")


async def _list_dependabot_alerts(client, kwargs: dict) -> dict:
    owner = kwargs.pop("owner", None)
    repo = kwargs.pop("repo", None)
    if not owner or not repo:
        return {
            "status": 400,
            "error": "Missing 'owner' or 'repo' parameter",
            "data": None,
        }
    response = await run_blocking(
        client.get_dependabot_alerts, owner=owner, repo=repo, **kwargs
    )
    return {
        "status": 200,
        "message": "Dependabot alerts retrieved successfully",
        "data": response.data,
    }


async def _get_dependabot_alert(client, kwargs: dict) -> dict:
    owner = kwargs.get("owner")
    repo = kwargs.get("repo")
    alert_number = kwargs.get("alert_number")
    if not owner or not repo or not alert_number:
        return {
            "status": 400,
            "error": "Missing 'owner', 'repo', or 'alert_number' parameter",
            "data": None,
        }
    response = await run_blocking(
        client.get_dependabot_alert,
        owner=owner,
        repo=repo,
        alert_number=int(alert_number),
    )
    return {
        "status": 200,
        "message": "Dependabot alert retrieved successfully",
        "data": response.data,
    }


async def _list_org_dependabot_alerts(client, kwargs: dict) -> dict:
    org = kwargs.pop("org", None)
    if not org:
        return {
            "status": 400,
            "error": "Missing required 'org' parameter",
            "data": None,
        }
    response = await run_blocking(client.get_org_dependabot_alerts, org=org, **kwargs)
    return {
        "status": 200,
        "message": "Organization Dependabot alerts retrieved successfully",
        "data": response.data,
    }


async def _update_dependabot_alert(client, kwargs: dict) -> dict:
    owner = kwargs.get("owner")
    repo = kwargs.get("repo")
    alert_number = kwargs.get("alert_number")
    state = kwargs.get("state")
    if not owner or not repo or not alert_number or not state:
        return {
            "status": 400,
            "error": "Missing 'owner', 'repo', 'alert_number', or 'state' parameter",
            "data": None,
        }
    response = await run_blocking(
        client.update_dependabot_alert,
        owner=owner,
        repo=repo,
        alert_number=int(alert_number),
        state=state,
        dismissed_reason=kwargs.get("dismissed_reason"),
        dismissed_comment=kwargs.get("dismissed_comment"),
    )
    return {
        "status": 200,
        "message": "Dependabot alert updated successfully",
        "data": response.data,
    }


#: Dispatch table for the resolved Dependabot action -> its async handler.
_DEPENDABOT_ACTION_HANDLERS = {
    "list": _list_dependabot_alerts,
    "get": _get_dependabot_alert,
    "list_org": _list_org_dependabot_alerts,
    "update": _update_dependabot_alert,
}


def _parse_dependabot_params(params_json: str) -> tuple[dict, None] | tuple[None, dict]:
    """Parse ``params_json`` and strip ``None`` values. Returns ``(kwargs, None)`` on
    success or ``(None, error_response)`` on a malformed payload."""
    import json

    try:
        raw = json.loads(params_json)
    except Exception as e:
        return None, {
            "status": 400,
            "error": f"Invalid params_json: {type(e).__name__}",
            "data": None,
        }
    return {k: v for k, v in raw.items() if v is not None}, None


def _dependabot_destructive_guard(action: str, allow_destructive: bool) -> dict | None:
    """Return a 403 error response if ``action`` is a blocked guarded write, else ``None``."""
    if action not in DESTRUCTIVE_DEPENDABOT_ACTIONS:
        return None
    if allow_destructive is True or allow_destructive_default():
        return None
    return {
        "status": 403,
        "error": (
            f"Action '{action}' is a guarded write and blocked by default. "
            "Re-run with allow_destructive=true (or set "
            "GITHUB_ALLOW_DESTRUCTIVE=True) to confirm."
        ),
        "data": None,
    }


def register_dependabot_tools(mcp: FastMCP):
    @mcp.tool(tags={"dependabot"})
    async def github_dependabot(
        action: str = Field(
            description="Action to perform. Must be one of: 'list', 'get', 'list_org', 'update'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        allow_destructive: bool = Field(
            default=False,
            description="Confirm a guarded write ('update' — dismiss/reopen an alert). Also honoured via GITHUB_ALLOW_DESTRUCTIVE.",
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Review and manage GitHub Dependabot vulnerability alerts.

        Actions (parameters via params_json):
        - 'list': {"owner", "repo", plus optional filters: state
          (open/dismissed/fixed/auto_dismissed), severity
          (low/medium/high/critical), ecosystem, package, scope
          (development/runtime), sort, direction, per_page} — list a repo's
          Dependabot alerts (GET /repos/{owner}/{repo}/dependabot/alerts).
        - 'get': {"owner", "repo", "alert_number"} — a single alert.
        - 'list_org': {"org", plus the same optional filters} — every
          Dependabot alert across an organization
          (GET /orgs/{org}/dependabot/alerts).
        - 'update': {"owner", "repo", "alert_number", "state":
          "dismissed"|"open", plus "dismissed_reason" (required to dismiss:
          fix_started/inaccurate/no_bandwidth/not_used/tolerable_risk) and
          optional "dismissed_comment"} — dismiss or reopen an alert. Guarded
          by allow_destructive / GITHUB_ALLOW_DESTRUCTIVE.
        """
        if ctx:
            await ctx.info("Executing github_dependabot action...")

        kwargs, parse_error = _parse_dependabot_params(params_json)
        if parse_error is not None:
            return parse_error

        resolved = resolve_action(action, DEPENDABOT_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        guard = _dependabot_destructive_guard(action, allow_destructive)
        if guard is not None:
            return guard

        handler = _DEPENDABOT_ACTION_HANDLERS.get(action)
        if handler is None:
            return {"status": 400, "error": f"Unknown action: {action}", "data": None}

        try:
            return await handler(client, kwargs)
        except Exception as e:
            return {"status": 500, "error": str(e), "data": None}
