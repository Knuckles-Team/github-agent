"""MCP tools for org operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp_utilities import resolve_action, run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.api.api_client_orgs import OrganizationCreationNotSupportedError
from github_agent.auth import allow_destructive_default, get_client

#: Actions gated behind allow_destructive / GITHUB_ALLOW_DESTRUCTIVE.
DESTRUCTIVE_ORG_ACTIONS = {"delete", "remove_member"}

#: Valid org actions for the shared ``resolve_action`` discovery helper.
ORG_ACTIONS = (
    "get",
    "list",
    "update",
    "delete",
    "create",
    "create_repository",
    "repos",
    "members",
    "get_membership",
    "set_membership",
    "remove_member",
    "teams",
)


def register_org_tools(mcp: FastMCP):
    @mcp.tool(tags={"orgs"})
    async def github_orgs(
        action: str = Field(
            description=(
                "Action to perform. Must be one of: 'get', 'list', 'update', "
                "'delete', 'create', 'create_repository', 'repos', 'members', "
                "'get_membership', 'set_membership', 'remove_member', 'teams'"
            )
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        allow_destructive: bool = Field(
            default=False,
            description=(
                "Must be true to run destructive actions: ['delete', "
                "'remove_member']. Organization deletion is IRREVERSIBLE."
            ),
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub organizations.

        Actions (parameters via params_json):
        - 'get': {"org"} — full organization profile (GET /orgs/{org}).
        - 'list': {"scope": "member"|"all", "since"} — scope 'member'
          (default) lists the authenticated user's organizations
          (GET /user/orgs); scope 'all' lists every organization
          (GET /organizations, paginated by the 'since' ID cursor).
        - 'update': {"org", plus mutable fields: billing_email, company,
          email, location, name, description, blog, twitter_username,
          has_organization_projects, has_repository_projects,
          default_repository_permission, members_can_create_repositories,
          web_commit_signoff_required, ...} — PATCH /orgs/{org}.
        - 'delete': {"org"} — IRREVERSIBLE; schedules organization deletion
          (HTTP 202). Requires allow_destructive=true.
        - 'create': {"login", "admin", "profile_name"} — GitHub Enterprise
          Server ONLY (POST /admin/organizations). github.com organizations
          CANNOT be created via the API (web UI only); against
          api.github.com this action returns an error explaining that.
        - 'create_repository': {"org", "name", plus the same payload fields
          as the github_repos 'create' action} — POST /orgs/{org}/repos.
        - 'repos': {"org", "type"} — list organization repositories.
        - 'members': {"org", "role"} — list organization members.
        - 'get_membership': {"org", "username"} — a user's membership state
          and role.
        - 'set_membership': {"org", "username", "role"} — add or update a
          member ('member' or 'admin'); invites the user if not yet a member.
        - 'remove_member': {"org", "username"} — remove a member; requires
          allow_destructive=true.
        - 'teams': {"org"} — list organization teams.
        """
        if ctx:
            await ctx.info("Executing github_orgs action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        resolved = resolve_action(action, ORG_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action in DESTRUCTIVE_ORG_ACTIONS and not (
            allow_destructive is True or allow_destructive_default()
        ):
            return {
                "status": 403,
                "error": (
                    f"Action '{action}' is destructive and blocked by default. "
                    "Re-run with allow_destructive=true (or set "
                    "GITHUB_ALLOW_DESTRUCTIVE=True) to confirm."
                ),
                "data": None,
            }

        try:
            if action == "get":
                org = kwargs.get("org")
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = await run_blocking(client.get_organization, org=org)
                return {
                    "status": 200,
                    "message": "Organization retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "list":
                response = await run_blocking(client.list_organizations, **kwargs)
                return {
                    "status": 200,
                    "message": "Organizations retrieved successfully",
                    "data": [org.model_dump() for org in response.data],
                }
            elif action == "update":
                org = kwargs.pop("org", None)
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.update_organization, org=org, **kwargs
                )
                return {
                    "status": 200,
                    "message": "Organization updated successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "delete":
                org = kwargs.get("org")
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = await run_blocking(client.delete_organization, org=org)
                return {
                    "status": 202,
                    "message": "Organization deletion scheduled (irreversible)",
                    "data": response.data,
                }
            elif action == "create":
                login = kwargs.get("login")
                admin = kwargs.get("admin")
                if not login or not admin:
                    return {
                        "status": 400,
                        "error": "Missing 'login' or 'admin' parameter",
                        "data": None,
                    }
                try:
                    response = await run_blocking(
                        client.create_organization,
                        login=login,
                        admin=admin,
                        profile_name=kwargs.get("profile_name"),
                    )
                except OrganizationCreationNotSupportedError as e:
                    return {"status": 400, "error": str(e), "data": None}
                return {
                    "status": 201,
                    "message": "Organization created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "create_repository":
                org = kwargs.pop("org", None)
                name = kwargs.pop("name", None)
                if not org or not name:
                    return {
                        "status": 400,
                        "error": "Missing 'org' or 'name' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.create_organization_repository, org=org, name=name, **kwargs
                )
                return {
                    "status": 201,
                    "message": "Organization repository created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "repos":
                response = await run_blocking(client.get_org_repos, **kwargs)
                return {
                    "status": 200,
                    "message": "Organization repositories retrieved successfully",
                    "data": [repo.model_dump() for repo in response.data],
                }
            elif action == "members":
                response = await run_blocking(client.get_org_members, **kwargs)
                return {
                    "status": 200,
                    "message": "Organization members retrieved successfully",
                    "data": [member.model_dump() for member in response.data],
                }
            elif action == "get_membership":
                org = kwargs.get("org")
                username = kwargs.get("username")
                if not org or not username:
                    return {
                        "status": 400,
                        "error": "Missing 'org' or 'username' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_organization_membership, org=org, username=username
                )
                return {
                    "status": 200,
                    "message": "Organization membership retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "set_membership":
                org = kwargs.get("org")
                username = kwargs.get("username")
                if not org or not username:
                    return {
                        "status": 400,
                        "error": "Missing 'org' or 'username' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.set_organization_membership,
                    org=org,
                    username=username,
                    role=kwargs.get("role", "member"),
                )
                return {
                    "status": 200,
                    "message": "Organization membership set successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "remove_member":
                org = kwargs.get("org")
                username = kwargs.get("username")
                if not org or not username:
                    return {
                        "status": 400,
                        "error": "Missing 'org' or 'username' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.remove_organization_member, org=org, username=username
                )
                return {
                    "status": 200,
                    "message": "Organization member removed successfully",
                    "data": response.data,
                }
            elif action == "teams":
                org = kwargs.get("org")
                if not org:
                    return {
                        "status": 400,
                        "error": "Missing required 'org' parameter",
                        "data": None,
                    }
                response = await run_blocking(client.get_org_teams, org=org)
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
