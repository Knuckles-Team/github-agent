"""Action-discovery contract for the github-agent MCP tools.

Every action-routed tool dispatches through the shared
``agent_connector_sdk.mcp.action_dispatch.resolve_action`` helper, which gives callers
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


# --------------------------------------------------------------------------
# BUG-CX-035: unreachable "Unknown action" fallback (11 residual sites)
# --------------------------------------------------------------------------
#
# Every action-routed tool in mcp_server.py resolves `action` via
# resolve_action(action, XXX_ACTIONS, ...) before ever touching a handler
# dict/if-elif. resolve_action() either raises ValueError for anything not
# canonicalizable to a member of XXX_ACTIONS, or returns a canonical member
# of it -- so a subsequent `_XXX_ACTION_HANDLERS.get(action)` (or an
# if/elif over the same set) can never miss. The `if handler is None: return
# {"error": "Unknown action: ..."}` / `else: return {...}` fallback that
# used to follow was therefore dead code the real input space could never
# reach. Two sites (github_repos, github_comments) were fixed under
# BUG-CX-035 in an earlier lane (WD3-FL-05); the other 11 residual copies
# are fixed here (lane WD10-B-SMALL).


def test_no_reachable_unknown_action_fallback_remains():
    """The literal dead-fallback string must be gone from mcp_server.py.

    Fails against unmodified main (11 residual occurrences); passes once
    every site is converted to direct indexing / an exhaustive if/elif.
    """
    import inspect

    from github_agent import mcp_server

    source = inspect.getsource(mcp_server)
    assert 'f"Unknown action: {action}"' not in source


# (ACTIONS tuple name, handler dict name) pairs for every dict-dispatched
# action-routed tool in mcp_server.py, including the two already fixed under
# BUG-CX-035 in an earlier lane.
_ACTION_HANDLER_PAIRS = [
    ("REPO_ACTIONS", "_REPO_ACTION_HANDLERS"),
    ("COMMENT_ACTIONS", "_COMMENT_ACTION_HANDLERS"),
    ("ISSUE_ACTIONS", "_ISSUE_ACTION_HANDLERS"),
    ("PULL_ACTIONS", "_PULL_ACTION_HANDLERS"),
    ("CONTENT_ACTIONS", "_CONTENT_ACTION_HANDLERS"),
    ("BRANCH_ACTIONS", "_BRANCH_ACTION_HANDLERS"),
    ("COMMIT_ACTIONS", "_COMMIT_ACTION_HANDLERS"),
    ("ORG_ACTIONS", "_ORG_ACTION_HANDLERS"),
    ("COLLABORATOR_ACTIONS", "_COLLABORATOR_ACTION_HANDLERS"),
    ("WORKFLOW_ACTIONS", "_ACTION_HANDLERS"),
    ("RELEASE_ACTIONS", "_RELEASE_ACTION_HANDLERS"),
    ("DEPENDABOT_ACTIONS", "_DEPENDABOT_ACTION_HANDLERS"),
]


@pytest.mark.parametrize("actions_name,handlers_name", _ACTION_HANDLER_PAIRS)
def test_action_tuple_matches_handler_dict_keys(actions_name, handlers_name):
    """The invariant BUG-CX-035's fix depends on: every dispatch dict's key
    set is identical to its paired ACTIONS tuple, so resolve_action() having
    already validated `action` against the tuple guarantees the dict lookup
    can never miss."""
    from github_agent import mcp_server

    actions = set(getattr(mcp_server, actions_name))
    handlers = set(getattr(mcp_server, handlers_name).keys())
    assert actions == handlers


def test_search_actions_matches_the_if_elif_branches():
    """github_search dispatches via an if/elif chain (not a dict), so the
    same invariant is pinned directly against its three literal branches."""
    from github_agent import mcp_server

    assert set(mcp_server.SEARCH_ACTIONS) == {"repositories", "issues", "code"}
