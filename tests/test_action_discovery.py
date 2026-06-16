"""Action-discovery contract for the github-agent MCP tools.

Every action-routed tool dispatches through the shared
``agent_utilities.mcp_utilities.resolve_action`` helper, which gives callers
``list_actions`` discovery and a rich did-you-mean error on an unknown action.
These tests assert that contract on the live tool dispatch path.
"""

import inspect
from unittest.mock import MagicMock

import pytest


async def _registered_tools():
    from github_agent.mcp_server import get_mcp_instance

    mcp = get_mcp_instance()[0]
    if inspect.iscoroutinefunction(mcp.list_tools):
        tools = await mcp.list_tools()
    else:
        tools = mcp.list_tools()
    return {t.name: t.fn for t in tools}


# (tool name, an action that is NOT valid for that tool)
_TOOLS = [
    "github_repos",
    "github_issues",
    "github_pulls",
    "github_contents",
    "github_branches",
    "github_commits",
    "github_search",
    "github_orgs",
    "github_collaborators",
    "github_actions",
    "github_releases",
]


@pytest.mark.parametrize("tool_name", _TOOLS)
async def test_list_actions_returns_names(tool_name):
    tools = await _registered_tools()
    tool = tools[tool_name]
    result = await tool(
        action="list_actions",
        params_json="{}",
        client=MagicMock(),
        ctx=None,
    )
    assert isinstance(result, dict)
    assert result["service"] == "github-agent"
    assert isinstance(result["actions"], list) and result["actions"]


@pytest.mark.parametrize("tool_name", _TOOLS)
async def test_unknown_action_raises_did_you_mean(tool_name):
    tools = await _registered_tools()
    tool = tools[tool_name]
    with pytest.raises(ValueError, match="list_actions"):
        await tool(
            action="definitely_not_an_action",
            params_json="{}",
            client=MagicMock(),
            ctx=None,
        )


async def test_plural_alias_resolves_to_singular():
    """An intuitive plural (get_runs) resolves to the singular method (get_run)."""
    tools = await _registered_tools()
    tool = tools["github_actions"]
    client = MagicMock()
    client.get_workflow_run.return_value = MagicMock(data=MagicMock())
    # WORKFLOW_ACTIONS contains 'get_run'; calling 'get_runs' must alias to it
    # rather than raising, so the dispatch reaches the get_run branch.
    result = await tool(
        action="get_runs",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=None,
    )
    assert isinstance(result, dict)
    assert result.get("status") != 400 or "Unknown action" not in str(
        result.get("error", "")
    )
