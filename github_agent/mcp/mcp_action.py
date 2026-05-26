"""MCP tools for action operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from github_agent.auth import get_client


def register_action_tools(mcp: FastMCP):
    @mcp.tool(tags={"actions"})
    async def github_actions(
        action: str = Field(
            description="Action to perform. Must be one of: 'list_workflows', 'list_runs', 'get_run', 'trigger_dispatch', 'rerun', 'cancel', 'delete_run'"
        ),
        params_json: str = Field(
            default="{}", description="JSON string of parameters to pass to the action."
        ),
        client=Depends(get_client),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage GitHub actions workflows and runs."""
        if ctx:
            await ctx.info("Executing github_actions action...")
        import json

        try:
            kwargs = json.loads(params_json)
        except Exception as e:
            return {"status": 400, "error": f"Invalid params_json: {e}", "data": None}

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            if action == "list_workflows":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                if not owner or not repo:
                    return {
                        "status": 400,
                        "error": "Missing required 'owner' or 'repo' parameter",
                        "data": None,
                    }
                response = client.get_workflows(owner=owner, repo=repo)
                return {
                    "status": 200,
                    "message": "Workflows retrieved successfully",
                    "data": [w.model_dump() for w in response.data],
                }
            elif action == "list_runs":
                response = client.get_workflow_runs(**kwargs)
                return {
                    "status": 200,
                    "message": "Workflow runs retrieved successfully",
                    "data": [r.model_dump() for r in response.data],
                }
            elif action == "get_run":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                run_id = kwargs.get("run_id")
                if not owner or not repo or not run_id:
                    return {
                        "status": 400,
                        "error": "Missing required 'owner', 'repo', or 'run_id' parameter",
                        "data": None,
                    }
                response = client.get_workflow_run(
                    owner=owner, repo=repo, run_id=int(run_id)
                )
                return {
                    "status": 200,
                    "message": "Workflow run retrieved successfully",
                    "data": response.data.model_dump(),
                }
            elif action == "trigger_dispatch":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                workflow_id = kwargs.get("workflow_id")
                ref = kwargs.get("ref")
                inputs = kwargs.get("inputs")
                if not owner or not repo or not workflow_id or not ref:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', 'workflow_id', or 'ref' parameter",
                        "data": None,
                    }
                response = client.trigger_workflow_dispatch(
                    owner=owner,
                    repo=repo,
                    workflow_id=workflow_id,
                    ref=ref,
                    inputs=inputs,
                )
                return {
                    "status": 200,
                    "message": "Workflow dispatch triggered successfully",
                    "data": response.data,
                }
            elif action == "rerun":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                run_id = kwargs.get("run_id")
                if not owner or not repo or not run_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'run_id' parameter",
                        "data": None,
                    }
                response = client.rerun_workflow_run(
                    owner=owner, repo=repo, run_id=int(run_id)
                )
                return {
                    "status": 200,
                    "message": "Workflow run rerun triggered",
                    "data": response.data,
                }
            elif action == "cancel":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                run_id = kwargs.get("run_id")
                if not owner or not repo or not run_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'run_id' parameter",
                        "data": None,
                    }
                response = client.cancel_workflow_run(
                    owner=owner, repo=repo, run_id=int(run_id)
                )
                return {
                    "status": 200,
                    "message": "Workflow run cancellation triggered",
                    "data": response.data,
                }
            elif action == "delete_run":
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                run_id = kwargs.get("run_id")
                if not owner or not repo or not run_id:
                    return {
                        "status": 400,
                        "error": "Missing 'owner', 'repo', or 'run_id' parameter",
                        "data": None,
                    }
                response = client.delete_workflow_run(
                    owner=owner, repo=repo, run_id=int(run_id)
                )
                return {
                    "status": 200,
                    "message": "Workflow run deleted successfully",
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
