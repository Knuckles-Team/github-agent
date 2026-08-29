"""Direct dispatch coverage for the ``github_agent.mcp`` package mirror.

``github_agent/mcp/mcp_*.py`` is a parallel, standalone copy of the tool
registrars also defined inline in ``github_agent/mcp_server.py`` (the actual
``github-mcp`` entrypoint's ``TOOL_REGISTRY`` uses ONLY the ``mcp_server.py``
copies — see BUGS FOUND in the WD11-FL-04 lane report). ``tests/test_mcp_coverage.py``'s
``get_registered_tools()`` goes through ``mcp_server.get_mcp_instance()``, so its
tests exercise the ``mcp_server.py`` copies, never the ``github_agent/mcp/`` mirror.
Only two prior tests (``test_mcp_repo_module_mirror_byte_parity`` /
``test_mcp_orgs_module_mirror_registers_same_actions``) touch the mirror package
at all, and neither exercises its dispatch behaviour.

This file directly imports each ``register_*_tools`` from ``github_agent.mcp``
and drives its registered tool the same way ``test_mcp_coverage.py`` drives the
``mcp_server.py`` copies, so the mirror package's own action branches, guarded
writes, and error paths are actually characterized.
"""

from __future__ import annotations

import inspect

import pytest
from fastmcp import FastMCP

from github_agent.api.api_client_orgs import OrganizationCreationNotSupportedError
from tests.test_mcp_coverage import AsyncMockContext, create_mock_client


async def _register_and_get_tool(register_fn, tool_name: str):
    mcp = FastMCP("mirror-dispatch-check")
    register_fn(mcp)
    tools = (
        await mcp.list_tools()
        if inspect.iscoroutinefunction(mcp.list_tools)
        else mcp.list_tools()
    )
    return {t.name: t.fn for t in tools}[tool_name]


@pytest.mark.anyio
async def test_mirror_actions_dispatch():
    from github_agent.mcp.mcp_action import register_action_tools

    github_actions = await _register_and_get_tool(register_action_tools, "github_actions")
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_actions(
        action="list_workflows",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="list_workflows", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_actions(action="list_runs", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_actions(
        action="get_run",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(action="get_run", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_actions(
        action="list_jobs",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    # default slim=True strips *_url (keeps html_url).
    assert "check_run_url" not in res["data"][0]

    res = await github_actions(
        action="list_jobs",
        params_json='{"owner": "o", "repo": "r", "run_id": 1, "slim": false}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert "check_run_url" in res["data"][0]

    res = await github_actions(action="list_jobs", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_actions(
        action="job_logs",
        params_json='{"owner": "o", "repo": "r", "job_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(action="job_logs", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_actions(
        action="trigger_dispatch",
        params_json='{"owner": "o", "repo": "r", "workflow_id": 1, "ref": "m"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="trigger_dispatch", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_actions(
        action="rerun",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="cancel",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="delete_run",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    with pytest.raises(ValueError, match="list_actions"):
        await github_actions(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_commits_dispatch():
    from github_agent.mcp.mcp_commit import register_commit_tools

    github_commits = await _register_and_get_tool(register_commit_tools, "github_commits")
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_commits(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_commits(
        action="get",
        params_json='{"owner": "o", "repo": "r", "sha": "s"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_commits(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    with pytest.raises(ValueError, match="list_actions"):
        await github_commits(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_branches_dispatch():
    from github_agent.mcp.mcp_branch import register_branch_tools

    github_branches = await _register_and_get_tool(register_branch_tools, "github_branches")
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_branches(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_branches(
        action="get",
        params_json='{"owner": "o", "repo": "r", "branch": "b"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_branches(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_branches(
        action="create",
        params_json='{"owner": "o", "repo": "r", "branch": "b", "ref": "main"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_branches(action="create", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_branches(
        action="delete",
        params_json='{"owner": "o", "repo": "r", "branch": "b"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_branches(action="delete", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    # Declared-but-unimplemented actions (pre-existing, see BUGS FOUND).
    res = await github_branches(
        action="get_protection",
        params_json='{"owner": "o", "repo": "r", "branch": "b"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400
    assert "Unknown action" in res["error"]

    with pytest.raises(ValueError, match="list_actions"):
        await github_branches(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_collaborators_dispatch():
    from github_agent.mcp.mcp_collaborator import register_collaborator_tools

    github_collaborators = await _register_and_get_tool(
        register_collaborator_tools, "github_collaborators"
    )
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_collaborators(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_collaborators(
        action="add",
        params_json='{"owner": "o", "repo": "r", "username": "u"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_collaborators(action="add", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_collaborators(
        action="remove",
        params_json='{"owner": "o", "repo": "r", "username": "u"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_collaborators(action="remove", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    with pytest.raises(ValueError, match="list_actions"):
        await github_collaborators(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_contents_dispatch():
    from github_agent.mcp.mcp_content import register_content_tools

    github_contents = await _register_and_get_tool(register_content_tools, "github_contents")
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_contents(
        action="get", params_json='{"owner": "o", "repo": "r", "path": "p"}',
        client=client, ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_contents(
        action="create",
        params_json='{"owner": "o", "repo": "r", "path": "p", "message": "m", "content": "c"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_contents(action="create", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_contents(
        action="update",
        params_json=(
            '{"owner": "o", "repo": "r", "path": "p", "message": "m", '
            '"content": "c", "sha": "s"}'
        ),
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_contents(action="update", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_contents(
        action="delete",
        params_json='{"owner": "o", "repo": "r", "path": "p", "message": "m", "sha": "s"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_contents(action="delete", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    with pytest.raises(ValueError, match="list_actions"):
        await github_contents(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_dependabot_dispatch():
    from github_agent.mcp.mcp_dependabot import register_dependabot_tools

    github_dependabot = await _register_and_get_tool(
        register_dependabot_tools, "github_dependabot"
    )
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_dependabot(
        action="list", params_json='{"owner": "o", "repo": "r"}', client=client, ctx=ctx
    )
    assert res["status"] == 200

    res = await github_dependabot(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_dependabot(
        action="get",
        params_json='{"owner": "o", "repo": "r", "alert_number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_dependabot(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_dependabot(
        action="list_org", params_json='{"org": "acme"}', client=client, ctx=ctx
    )
    assert res["status"] == 200

    res = await github_dependabot(action="list_org", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_dependabot(
        action="update",
        params_json='{"owner": "o", "repo": "r", "alert_number": 1, "state": "dismissed"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403

    res = await github_dependabot(
        action="update",
        params_json='{"owner": "o", "repo": "r", "alert_number": 1, "state": "dismissed"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    with pytest.raises(ValueError, match="list_actions"):
        await github_dependabot(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_issues_dispatch():
    from github_agent.mcp.mcp_issue import register_issue_tools

    github_issues = await _register_and_get_tool(register_issue_tools, "github_issues")
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_issues(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_issues(
        action="get",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_issues(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_issues(
        action="create",
        params_json='{"owner": "o", "repo": "r", "title": "t"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_issues(action="create", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_issues(
        action="update",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_issues(action="update", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    # Org-wide search routing.
    client.reset_mock()
    res = await github_issues(
        action="list",
        params_json='{"org": "Knuckles-Team", "state": "open"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.search_issues.assert_called_once()
    client.get_issues.assert_not_called()

    with pytest.raises(ValueError, match="list_actions"):
        await github_issues(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_orgs_dispatch():
    from github_agent.mcp.mcp_org import register_org_tools

    github_orgs = await _register_and_get_tool(register_org_tools, "github_orgs")
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_orgs(action="get", params_json='{"org": "acme"}', client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_orgs(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_orgs(
        action="update",
        params_json='{"org": "acme", "company": "Acme"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_orgs(
        action="create_repository",
        params_json='{"org": "acme", "name": "repo"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_orgs(action="repos", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_orgs(action="members", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_orgs(
        action="get_membership",
        params_json='{"org": "acme", "username": "u"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_orgs(
        action="set_membership",
        params_json='{"org": "acme", "username": "u"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_orgs(
        action="teams", params_json='{"org": "acme"}', client=client, ctx=ctx
    )
    assert res["status"] == 200

    from unittest.mock import patch as _patch

    with _patch.object(
        client,
        "create_organization",
        side_effect=OrganizationCreationNotSupportedError("not supported"),
    ):
        res = await github_orgs(
            action="create",
            params_json='{"login": "acme", "admin": "u"}',
            client=client,
            ctx=ctx,
        )
    assert res["status"] == 400
    assert "not supported" in res["error"]

    res = await github_orgs(
        action="delete", params_json='{"org": "acme"}', client=client, ctx=ctx
    )
    assert res["status"] == 403

    res = await github_orgs(
        action="delete",
        params_json='{"org": "acme"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 202

    res = await github_orgs(
        action="remove_member",
        params_json='{"org": "acme", "username": "u"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    with pytest.raises(ValueError, match="list_actions"):
        await github_orgs(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_pulls_dispatch():
    from github_agent.mcp.mcp_pull import register_pull_tools

    github_pulls = await _register_and_get_tool(register_pull_tools, "github_pulls")
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_pulls(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_pulls(
        action="get",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_pulls(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_pulls(
        action="create",
        params_json='{"owner": "o", "repo": "r", "title": "t", "head": "h", "base": "b"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_pulls(action="create", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_pulls(
        action="update",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    client.create_pull_request_review = client.create_pull_request_review
    from unittest.mock import MagicMock

    client.create_pull_request_review = MagicMock(
        return_value=MagicMock(data={"state": "APPROVED"})
    )
    res = await github_pulls(
        action="approve",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    client.request_reviewers = MagicMock(
        return_value=MagicMock(data=MagicMock(model_dump=lambda: {"requested": ["u"]}))
    )
    res = await github_pulls(
        action="request_reviewers",
        params_json='{"owner": "o", "repo": "r", "number": 1, "reviewers": ["u"]}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    # 'merge' is a guarded write.
    res = await github_pulls(
        action="merge",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403

    client.merge_pull_request = MagicMock(return_value=MagicMock(data={"merged": True}))
    res = await github_pulls(
        action="merge",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    with pytest.raises(ValueError, match="list_actions"):
        await github_pulls(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_releases_dispatch():
    from github_agent.mcp.mcp_release import register_release_tools

    github_releases = await _register_and_get_tool(register_release_tools, "github_releases")
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_releases(
        action="list", params_json='{"owner": "o", "repo": "r"}', client=client, ctx=ctx
    )
    assert res["status"] == 200

    res = await github_releases(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_releases(
        action="get",
        params_json='{"owner": "o", "repo": "r", "release_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_releases(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_releases(
        action="create",
        params_json='{"owner": "o", "repo": "r", "tag_name": "v1"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_releases(action="create", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_releases(
        action="update",
        params_json='{"owner": "o", "repo": "r", "release_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_releases(action="update", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_releases(
        action="delete",
        params_json='{"owner": "o", "repo": "r", "release_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_releases(action="delete", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    with pytest.raises(ValueError, match="list_actions"):
        await github_releases(action="invalid", params_json="{}", client=client, ctx=ctx)


@pytest.mark.anyio
async def test_mirror_pulls_rest_actions_survive_gql_client_failure(monkeypatch):
    """Same regression guard as mcp_server.py's github_pulls, for the mirror."""
    from github_agent.mcp.mcp_pull import register_pull_tools

    github_pulls = await _register_and_get_tool(register_pull_tools, "github_pulls")
    client = create_mock_client()
    ctx = AsyncMockContext()

    def _broken_graphql_client(*args, **kwargs):
        raise RuntimeError("no graphql client available")

    monkeypatch.setattr(
        "github_agent.mcp.mcp_pull.get_graphql_client", _broken_graphql_client
    )

    res = await github_pulls(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_pulls(
        action="enable_auto_merge",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 500
    assert "GraphQL client unavailable" in res["error"]
