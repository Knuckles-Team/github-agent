"""MCP tools for repo operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import allow_destructive_default, get_client
from github_agent.github_response_models import PagesAlreadyEnabled, PagesNotEnabled

#: Repo actions gated behind allow_destructive / GITHUB_ALLOW_DESTRUCTIVE.
DESTRUCTIVE_REPO_ACTIONS = {"pages_delete", "secrets_delete"}

#: Valid repo actions for the shared ``resolve_action`` discovery helper.
REPO_ACTIONS = (
    "list",
    "get",
    "create",
    "delete",
    "update",
    "pages_get",
    "pages_create",
    "pages_update",
    "pages_delete",
    "pages_builds",
    "pages_request_build",
    "secrets_list",
    "secrets_public_key",
    "secrets_set",
    "secrets_delete",
)

#: Exact keys dropped by _slim (pure hypermedia/noise, never semantic data).
_SLIM_DROP_EXACT = {"_links", "url", "node_id"}


def _slim(obj):
    """Recursively drop hypermedia ``*_url`` hrefs and ``_links`` noise.

    Mirror of ``mcp_server._slim``; see that module for rationale.
    """
    if isinstance(obj, list):
        return [_slim(item) for item in obj]
    if isinstance(obj, dict):
        return {
            k: _slim(v)
            for k, v in obj.items()
            if k not in _SLIM_DROP_EXACT
            and not (k.endswith("_url") and k != "html_url")
        }
    return obj


def register_repo_tools(mcp: FastMCP):
    @mcp.tool(tags={"repos"})
    async def github_repos(
        action: str = Field(
            description=(
                "Action to perform. Must be one of: 'list', 'get', 'create', "
                "'delete', 'update', 'pages_get', 'pages_create', "
                "'pages_update', 'pages_delete', 'pages_builds', "
                "'pages_request_build', 'secrets_list', 'secrets_public_key', "
                "'secrets_set', 'secrets_delete'"
            )
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        allow_destructive: bool = Field(
            default=False,
            description=(
                "Must be true to run destructive actions: ['pages_delete', "
                "'secrets_delete']. Deleting a Pages site takes it offline "
                "immediately; deleting a secret removes it permanently."
            ),
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub repositories and their GitHub Pages sites.

        Repository actions (parameters via params_json):
        - 'list': {"visibility", "affiliation", "type"} — repositories for
          the authenticated user.
        - 'get': {"owner", "repo"} — a single repository.
        - 'create': {"name", plus payload fields such as description,
          private, auto_init, ...} — create a repository for the
          authenticated user.
        - 'delete': {"owner", "repo"} — delete a repository.
        - 'update': {"owner", "repo", plus mutable repository settings} —
          PATCH /repos/{owner}/{repo}.

        GitHub Pages actions:
        - 'pages_get': {"owner", "repo"} — the Pages site configuration;
          returns status 404 with a typed not-enabled result when Pages is
          off.
        - 'pages_create': {"owner", "repo", "build_type":
          "workflow"|"legacy", "source": {"branch", "path"}} — enable
          Pages. build_type defaults to 'workflow'; 'legacy' requires
          source. Returns status 409 with a typed already-enabled result
          when Pages is already on.
        - 'pages_update': {"owner", "repo", plus build_type, source, cname,
          https_enforced} — change the Pages configuration.
        - 'pages_delete': {"owner", "repo"} — disable Pages and delete the
          site; requires allow_destructive=true.
        - 'pages_builds': {"owner", "repo", "latest": true|false} — list
          Pages builds, or only the latest build with latest=true.
        - 'pages_request_build': {"owner", "repo"} — request a fresh Pages
          build without pushing a commit (the programmatic fix for the
          first-deploy race where the initial Pages build never ran).

        Actions secrets actions:
        - 'secrets_list': {"owner", "repo"} — list repository Actions secret
          names (values are never returned by GitHub).
        - 'secrets_public_key': {"owner", "repo"} — the repository public key
          used to encrypt secret values before uploading them.
        - 'secrets_set': {"owner", "repo", "secret_name", "encrypted_value",
          "key_id"} — create or update a secret; encrypted_value must be
          sealed with the public key from 'secrets_public_key'.
        - 'secrets_delete': {"owner", "repo", "secret_name"} — permanently
          delete a secret; requires allow_destructive=true.
        """
        if ctx:
            await ctx.info("Executing github_repos action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {type(e).__name__}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        slim = kwargs.pop("slim", True)

        resolved = resolve_action(action, REPO_ACTIONS, service="github-agent")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action in DESTRUCTIVE_REPO_ACTIONS and not (
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
            if action == "list":
                response = await run_blocking(client.get_repositories, **kwargs)
                data = [repo.model_dump() for repo in response.data]
                return {
                    "status": 200,
                    "message": "Repositories retrieved successfully",
                    "data": _slim(data) if slim else data,
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
                response = await run_blocking(
                    client.get_repository, owner=owner, repo=repo
                )
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
                response = await run_blocking(
                    client.create_repository, name=name, **kwargs
                )
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
                response = await run_blocking(
                    client.delete_repository, owner=owner, repo=repo
                )
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
                response = await run_blocking(
                    client.update_repository, owner=owner, repo=repo, **kwargs
                )
                return {
                    "status": 200,
                    "message": "Repository updated successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "pages_get":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(client.get_pages, owner=owner, repo=repo)
                if isinstance(response.data, PagesNotEnabled):
                    return {
                        "status": 404,
                        "message": "GitHub Pages is not enabled for this repository",
                        "data": response.data.model_dump(),
                    }
                return {
                    "status": 200,
                    "message": "Pages site retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "pages_create":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.create_pages,
                    owner=owner,
                    repo=repo,
                    build_type=kwargs.get("build_type", "workflow"),
                    source=kwargs.get("source"),
                )
                if isinstance(response.data, PagesAlreadyEnabled):
                    return {
                        "status": 409,
                        "message": "GitHub Pages is already enabled for this repository",
                        "data": response.data.model_dump(),
                    }
                return {
                    "status": 201,
                    "message": "Pages site created successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "pages_update":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.update_pages, owner=owner, repo=repo, **kwargs
                )
                return {
                    "status": 200,
                    "message": "Pages site updated successfully",
                    "data": response.data,
                }
            elif action == "pages_delete":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.delete_pages, owner=owner, repo=repo
                )
                return {
                    "status": 200,
                    "message": "Pages site deleted successfully",
                    "data": response.data,
                }
            elif action == "pages_builds":
                owner = kwargs.pop("owner", None)
                repo = kwargs.pop("repo", None)
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                if kwargs.pop("latest", False):
                    response = await run_blocking(
                        client.get_pages_build_latest, owner=owner, repo=repo
                    )
                    return {
                        "status": 200,
                        "message": "Latest Pages build retrieved successfully",
                        "data": response.data.model_dump(),
                    }
                response = await run_blocking(
                    client.list_pages_builds, owner=owner, repo=repo, **kwargs
                )
                return {
                    "status": 200,
                    "message": "Pages builds retrieved successfully",
                    "data": [build.model_dump() for build in response.data],
                }
            elif action == "pages_request_build":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.request_pages_build, owner=owner, repo=repo
                )
                return {
                    "status": 201,
                    "message": "Pages build requested successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "secrets_list":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_repo_secrets, owner=owner, repo=repo
                )
                return {
                    "status": 200,
                    "message": "Repository secrets retrieved successfully",
                    "data": response.data,
                }
            elif action == "secrets_public_key":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.get_repo_secret_public_key, owner=owner, repo=repo
                )
                return {
                    "status": 200,
                    "message": "Repository secret public key retrieved successfully",
                    "data": response.data,
                }
            elif action == "secrets_set":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                secret_name = kwargs.get("secret_name")
                encrypted_value = kwargs.get("encrypted_value")
                key_id = kwargs.get("key_id")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                if not secret_name or encrypted_value is None or not key_id:
                    return {
                        "status": 400,
                        "error": (
                            "Missing 'secret_name', 'encrypted_value', or "
                            "'key_id' parameter"
                        ),
                        "data": None,
                    }
                response = await run_blocking(
                    client.create_or_update_repo_secret,
                    owner=owner,
                    repo=repo,
                    secret_name=secret_name,
                    encrypted_value=encrypted_value,
                    key_id=key_id,
                )
                return {
                    "status": 200,
                    "message": "Repository secret set successfully",
                    "data": response.data,
                }
            elif action == "secrets_delete":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                secret_name = kwargs.get("secret_name")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing 'owner' or 'repo' parameter",
                        "data": None,
                    }
                if not secret_name:
                    return {
                        "status": 400,
                        "error": "Missing 'secret_name' parameter",
                        "data": None,
                    }
                response = await run_blocking(
                    client.delete_repo_secret,
                    owner=owner,
                    repo=repo,
                    secret_name=secret_name,
                )
                return {
                    "status": 200,
                    "message": "Repository secret deleted successfully",
                    "data": response.data,
                }
            else:
                return {
                    "status": 400,
                    "error": f"Unknown action: {action}",
                    "data": None,
                }
        except Exception:
            return {"status": 500, "error": "Operation failed", "data": None}
