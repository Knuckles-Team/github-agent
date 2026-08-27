"""Characterization coverage for the github_repos 'secrets_*' actions.

CXA-FL-GITHUBAGENT-01: register_repo_tools.github_repos (mcp_server.py /
mcp/mcp_repo.py) had zero existing test coverage for its four
'secrets_list' / 'secrets_public_key' / 'secrets_set' / 'secrets_delete'
branches before this lane's refactor (grepped the full tests/ directory
for those four strings -- no hits). These tests pin their pre-refactor
behavior, including the destructive-gating and required-parameter paths,
so the table-driven refactor of `register_repo_tools` can be verified
byte-for-byte behavior preserving.
"""

import inspect

import pytest


class AsyncMockContext:
    def __init__(self):
        self.info_calls = []

    async def info(self, msg):
        self.info_calls.append(msg)


def make_secrets_client():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.get_repo_secrets.return_value = MagicMock(
        data={"total_count": 1, "secrets": [{"name": "TOKEN"}]}
    )
    client.get_repo_secret_public_key.return_value = MagicMock(
        data={"key_id": "1", "key": "abc"}
    )
    client.create_or_update_repo_secret.return_value = MagicMock(data={})
    client.delete_repo_secret.return_value = MagicMock(data={})
    return client


async def get_github_repos_tool():
    from github_agent.mcp_server import get_mcp_instance

    mcp = get_mcp_instance()[0]
    if inspect.iscoroutinefunction(mcp.list_tools):
        tools = await mcp.list_tools()
    else:
        tools = mcp.list_tools()
    return {t.name: t.fn for t in tools}["github_repos"]


@pytest.mark.anyio
async def test_mcp_repos_secrets_list():
    github_repos = await get_github_repos_tool()
    client = make_secrets_client()
    ctx = AsyncMockContext()

    res = await github_repos(
        action="secrets_list",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.get_repo_secrets.assert_called_with(owner="o", repo="r")

    res = await github_repos(
        action="secrets_list", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_repos_secrets_public_key():
    github_repos = await get_github_repos_tool()
    client = make_secrets_client()
    ctx = AsyncMockContext()

    res = await github_repos(
        action="secrets_public_key",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.get_repo_secret_public_key.assert_called_with(owner="o", repo="r")

    res = await github_repos(
        action="secrets_public_key", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_repos_secrets_set():
    github_repos = await get_github_repos_tool()
    client = make_secrets_client()
    ctx = AsyncMockContext()

    res = await github_repos(
        action="secrets_set",
        params_json=(
            '{"owner": "o", "repo": "r", "secret_name": "TOKEN", '
            '"encrypted_value": "enc", "key_id": "1"}'
        ),
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.create_or_update_repo_secret.assert_called_with(
        owner="o", repo="r", secret_name="TOKEN", encrypted_value="enc", key_id="1"
    )

    # missing owner/repo
    res = await github_repos(
        action="secrets_set",
        params_json='{"secret_name": "TOKEN", "encrypted_value": "enc", "key_id": "1"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400
    assert "owner" in res["error"] or "repo" in res["error"]

    # missing secret_name/encrypted_value/key_id
    res = await github_repos(
        action="secrets_set",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400
    assert "secret_name" in res["error"]


@pytest.mark.anyio
async def test_mcp_repos_secrets_delete_destructive_gating(monkeypatch):
    monkeypatch.delenv("GITHUB_ALLOW_DESTRUCTIVE", raising=False)
    github_repos = await get_github_repos_tool()
    client = make_secrets_client()
    ctx = AsyncMockContext()

    # blocked by default
    res = await github_repos(
        action="secrets_delete",
        params_json='{"owner": "o", "repo": "r", "secret_name": "TOKEN"}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403
    assert "allow_destructive" in res["error"]
    client.delete_repo_secret.assert_not_called()

    # allowed with explicit per-call consent
    res = await github_repos(
        action="secrets_delete",
        params_json='{"owner": "o", "repo": "r", "secret_name": "TOKEN"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.delete_repo_secret.assert_called_with(
        owner="o", repo="r", secret_name="TOKEN"
    )

    # allowed via the environment default
    monkeypatch.setenv("GITHUB_ALLOW_DESTRUCTIVE", "True")
    res = await github_repos(
        action="secrets_delete",
        params_json='{"owner": "o", "repo": "r", "secret_name": "TOKEN"}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    monkeypatch.delenv("GITHUB_ALLOW_DESTRUCTIVE", raising=False)

    # missing owner/repo (gate open)
    res = await github_repos(
        action="secrets_delete",
        params_json='{"secret_name": "TOKEN"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400

    # missing secret_name (gate open, owner/repo present)
    res = await github_repos(
        action="secrets_delete",
        params_json='{"owner": "o", "repo": "r"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400
    assert "secret_name" in res["error"]
