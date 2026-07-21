"""GitHub Pages coverage: API client methods + the github_repos MCP actions."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import requests
from agent_utilities.core.exceptions import ParameterError

from github_agent.api_client import Api
from github_agent.github_response_models import (
    PagesAlreadyEnabled,
    PagesBuild,
    PagesBuildRequest,
    PagesNotEnabled,
    PagesSite,
)

USER_PAYLOAD = {
    "login": "octocat",
    "id": 7,
    "node_id": "U_7",
    "avatar_url": "https://avatars.example.com/u/7",
    "url": "https://api.github.com/users/octocat",
    "html_url": "https://github.com/octocat",
    "type": "User",
    "site_admin": False,
}

PAGES_PAYLOAD = {
    "url": "https://api.github.com/repos/Knuckles-Team/service/pages",
    "status": "built",
    "cname": "docs.example.com",
    "custom_404": False,
    "html_url": "https://knuckles-team.github.io/service/",
    "build_type": "workflow",
    "source": {"branch": "main", "path": "/"},
    "public": True,
    "https_enforced": True,
}

BUILD_PAYLOAD = {
    "url": "https://api.github.com/repos/Knuckles-Team/service/pages/builds/1",
    "status": "built",
    "error": {"message": None},
    "pusher": USER_PAYLOAD,
    "commit": "abc123sha",
    "duration": 2104,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:01:00Z",
}

BUILD_REQUEST_PAYLOAD = {
    "url": "https://api.github.com/repos/Knuckles-Team/service/pages/builds/latest",
    "status": "queued",
}


def make_response(status_code=200, json_data=None, headers=None):
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = json_data if json_data is not None else {}
    response.text = "{}"
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=response
        )
    return response


@pytest.fixture
def mock_session():
    with patch("requests.Session") as mock_s:
        session = mock_s.return_value
        default = make_response()
        session.get.return_value = default
        session.post.return_value = default
        session.put.return_value = default
        session.delete.return_value = default
        session.patch.return_value = default
        yield session


# --- API client -------------------------------------------------------------


def test_get_pages(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=PAGES_PAYLOAD)

    res = api.get_pages(owner="Knuckles-Team", repo="service")

    assert isinstance(res.data, PagesSite)
    assert res.data.build_type == "workflow"
    assert res.data.cname == "docs.example.com"
    assert res.data.source is not None
    assert res.data.source.branch == "main"
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/service/pages",
        headers=api.headers,
    )


def test_get_pages_not_enabled_404(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(status_code=404)

    res = api.get_pages(owner="Knuckles-Team", repo="service")

    assert isinstance(res.data, PagesNotEnabled)
    assert res.data.enabled is False
    assert "not enabled" in res.data.message


def test_get_pages_other_http_error_raises(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(status_code=500)

    with pytest.raises(requests.exceptions.HTTPError):
        api.get_pages(owner="Knuckles-Team", repo="service")


def test_create_pages_workflow_default(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data=PAGES_PAYLOAD
    )

    res = api.create_pages(owner="Knuckles-Team", repo="service")

    assert isinstance(res.data, PagesSite)
    mock_session.post.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/service/pages",
        json={"build_type": "workflow"},
        headers=api.headers,
    )


def test_create_pages_legacy_with_source(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data=PAGES_PAYLOAD
    )

    # Mapping source with explicit path
    api.create_pages(
        owner="Knuckles-Team",
        repo="service",
        build_type="legacy",
        source={"branch": "gh-pages", "path": "/docs"},
    )
    assert mock_session.post.call_args.kwargs["json"] == {
        "build_type": "legacy",
        "source": {"branch": "gh-pages", "path": "/docs"},
    }

    # Branch name string normalizes to {'branch', 'path': '/'}
    api.create_pages(
        owner="Knuckles-Team", repo="service", build_type="legacy", source="gh-pages"
    )
    assert mock_session.post.call_args.kwargs["json"] == {
        "build_type": "legacy",
        "source": {"branch": "gh-pages", "path": "/"},
    }


def test_create_pages_legacy_requires_source(mock_session):
    api = Api(token="test")
    with pytest.raises(ParameterError):
        api.create_pages(owner="Knuckles-Team", repo="service", build_type="legacy")
    mock_session.post.assert_not_called()


def test_create_pages_invalid_build_type(mock_session):
    api = Api(token="test")
    with pytest.raises(ParameterError):
        api.create_pages(owner="Knuckles-Team", repo="service", build_type="magic")
    mock_session.post.assert_not_called()


def test_create_pages_invalid_source_shape(mock_session):
    api = Api(token="test")
    with pytest.raises(ParameterError):
        api.create_pages(
            owner="Knuckles-Team",
            repo="service",
            build_type="legacy",
            source={"path": "/docs"},
        )
    mock_session.post.assert_not_called()


def test_create_pages_conflict_already_enabled_409(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(status_code=409)

    res = api.create_pages(owner="Knuckles-Team", repo="service")

    assert isinstance(res.data, PagesAlreadyEnabled)
    assert res.data.already_enabled is True
    assert "already enabled" in res.data.message


def test_update_pages_field_passthrough(mock_session):
    api = Api(token="test")
    mock_session.put.return_value = make_response(status_code=204)

    res = api.update_pages(
        owner="Knuckles-Team",
        repo="service",
        build_type="legacy",
        source="gh-pages",
        cname="docs.example.com",
        https_enforced=True,
    )

    assert res.data == {"status": "pages_updated"}
    mock_session.put.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/service/pages",
        json={
            "build_type": "legacy",
            "source": {"branch": "gh-pages", "path": "/"},
            "cname": "docs.example.com",
            "https_enforced": True,
        },
        headers=api.headers,
    )


def test_update_pages_rejects_unknown_field(mock_session):
    api = Api(token="test")
    with pytest.raises(ParameterError):
        api.update_pages(owner="Knuckles-Team", repo="service", not_a_real_field="x")
    mock_session.put.assert_not_called()


def test_update_pages_requires_some_field(mock_session):
    api = Api(token="test")
    with pytest.raises(ParameterError):
        api.update_pages(owner="Knuckles-Team", repo="service")
    mock_session.put.assert_not_called()


def test_delete_pages_204(mock_session):
    api = Api(token="test")
    mock_session.delete.return_value = make_response(status_code=204)

    res = api.delete_pages(owner="Knuckles-Team", repo="service")

    assert res.data == {"status": "pages_deleted"}
    mock_session.delete.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/service/pages",
        headers=api.headers,
    )


def test_get_pages_build_latest(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=BUILD_PAYLOAD)

    res = api.get_pages_build_latest(owner="Knuckles-Team", repo="service")

    assert isinstance(res.data, PagesBuild)
    assert res.data.status == "built"
    assert res.data.commit == "abc123sha"
    call = mock_session.get.call_args
    assert (
        call.kwargs["url"]
        == "https://api.github.com/repos/Knuckles-Team/service/pages/builds/latest"
    )


def test_list_pages_builds(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[BUILD_PAYLOAD])

    res = api.list_pages_builds(owner="Knuckles-Team", repo="service")

    assert len(res.data) == 1
    assert isinstance(res.data[0], PagesBuild)
    call = mock_session.get.call_args
    assert (
        call.kwargs["url"]
        == "https://api.github.com/repos/Knuckles-Team/service/pages/builds"
    )


def test_request_pages_build(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data=BUILD_REQUEST_PAYLOAD
    )

    res = api.request_pages_build(owner="Knuckles-Team", repo="service")

    assert isinstance(res.data, PagesBuildRequest)
    assert res.data.status == "queued"
    mock_session.post.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/service/pages/builds",
        headers=api.headers,
    )


# --- MCP tool ----------------------------------------------------------------


class AsyncMockContext:
    def __init__(self):
        self.info_calls = []

    async def info(self, msg):
        self.info_calls.append(msg)


def make_pages_client():
    client = MagicMock()
    mock_site = MagicMock()
    mock_site.model_dump.return_value = dict(PAGES_PAYLOAD)
    client.get_pages.return_value = MagicMock(data=mock_site)
    client.create_pages.return_value = MagicMock(data=mock_site)
    client.update_pages.return_value = MagicMock(data={"status": "pages_updated"})
    client.delete_pages.return_value = MagicMock(data={"status": "pages_deleted"})
    mock_build = MagicMock()
    mock_build.model_dump.return_value = dict(BUILD_PAYLOAD)
    client.get_pages_build_latest.return_value = MagicMock(data=mock_build)
    client.list_pages_builds.return_value = MagicMock(data=[mock_build])
    mock_request = MagicMock()
    mock_request.model_dump.return_value = dict(BUILD_REQUEST_PAYLOAD)
    client.request_pages_build.return_value = MagicMock(data=mock_request)
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
async def test_mcp_repos_pages_get():
    github_repos = await get_github_repos_tool()
    client = make_pages_client()
    ctx = AsyncMockContext()

    res = await github_repos(
        action="pages_get",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"]["build_type"] == "workflow"
    client.get_pages.assert_called_with(owner="o", repo="r")

    # Typed not-enabled result surfaces as a structured 404
    client.get_pages.return_value = MagicMock(data=PagesNotEnabled())
    res = await github_repos(
        action="pages_get",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 404
    assert res["data"]["enabled"] is False
    assert "not enabled" in res["message"]

    res = await github_repos(
        action="pages_get", params_json='{"owner": "o"}', client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_repos_pages_create():
    github_repos = await get_github_repos_tool()
    client = make_pages_client()
    ctx = AsyncMockContext()

    res = await github_repos(
        action="pages_create",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    client.create_pages.assert_called_with(
        owner="o", repo="r", build_type="workflow", source=None
    )

    res = await github_repos(
        action="pages_create",
        params_json=(
            '{"owner": "o", "repo": "r", "build_type": "legacy", '
            '"source": {"branch": "gh-pages", "path": "/docs"}}'
        ),
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    client.create_pages.assert_called_with(
        owner="o",
        repo="r",
        build_type="legacy",
        source={"branch": "gh-pages", "path": "/docs"},
    )

    # Typed already-enabled result surfaces as a structured 409
    client.create_pages.return_value = MagicMock(data=PagesAlreadyEnabled())
    res = await github_repos(
        action="pages_create",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 409
    assert res["data"]["already_enabled"] is True
    assert "already enabled" in res["message"]

    res = await github_repos(
        action="pages_create", params_json='{"repo": "r"}', client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_repos_pages_update():
    github_repos = await get_github_repos_tool()
    client = make_pages_client()
    ctx = AsyncMockContext()

    res = await github_repos(
        action="pages_update",
        params_json=(
            '{"owner": "o", "repo": "r", "cname": "docs.example.com", '
            '"https_enforced": true}'
        ),
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"] == {"status": "pages_updated"}
    client.update_pages.assert_called_with(
        owner="o", repo="r", cname="docs.example.com", https_enforced=True
    )

    res = await github_repos(
        action="pages_update", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_repos_pages_delete_destructive_gating(monkeypatch):
    monkeypatch.delenv("GITHUB_ALLOW_DESTRUCTIVE", raising=False)
    github_repos = await get_github_repos_tool()
    client = make_pages_client()
    ctx = AsyncMockContext()

    # Blocked by default
    res = await github_repos(
        action="pages_delete",
        params_json='{"owner": "o", "repo": "r"}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403
    assert "allow_destructive" in res["error"]
    client.delete_pages.assert_not_called()

    # Allowed with explicit per-call consent
    res = await github_repos(
        action="pages_delete",
        params_json='{"owner": "o", "repo": "r"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"] == {"status": "pages_deleted"}

    # Allowed via the environment default
    monkeypatch.setenv("GITHUB_ALLOW_DESTRUCTIVE", "True")
    res = await github_repos(
        action="pages_delete",
        params_json='{"owner": "o", "repo": "r"}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    # Missing params still 400 (with gate open)
    res = await github_repos(
        action="pages_delete",
        params_json='{"owner": "o"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_repos_pages_builds():
    github_repos = await get_github_repos_tool()
    client = make_pages_client()
    ctx = AsyncMockContext()

    res = await github_repos(
        action="pages_builds",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert isinstance(res["data"], list)
    assert res["data"][0]["status"] == "built"
    client.list_pages_builds.assert_called_with(owner="o", repo="r")

    res = await github_repos(
        action="pages_builds",
        params_json='{"owner": "o", "repo": "r", "latest": true}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"]["status"] == "built"
    client.get_pages_build_latest.assert_called_with(owner="o", repo="r")

    res = await github_repos(
        action="pages_builds", params_json='{"repo": "r"}', client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_repos_pages_request_build():
    github_repos = await get_github_repos_tool()
    client = make_pages_client()
    ctx = AsyncMockContext()

    res = await github_repos(
        action="pages_request_build",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    assert res["data"]["status"] == "queued"
    client.request_pages_build.assert_called_with(owner="o", repo="r")

    res = await github_repos(
        action="pages_request_build", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_repo_module_mirror_registers_same_actions():
    """The mcp package mirror must expose the same repo tool as mcp_server."""
    from fastmcp import FastMCP

    from github_agent.mcp.mcp_repo import register_repo_tools

    mcp = FastMCP("mirror-check")
    register_repo_tools(mcp)
    if inspect.iscoroutinefunction(mcp.list_tools):
        tools = await mcp.list_tools()
    else:
        tools = mcp.list_tools()
    tool = {t.name: t for t in tools}["github_repos"]
    description = tool.description or ""
    for action in (
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
    ):
        assert f"'{action}'" in description


def test_mcp_repo_module_mirror_byte_parity():
    """register_repo_tools must be byte-identical in mcp_server.py and the
    mcp/ package mirror."""
    import github_agent.mcp.mcp_repo as mirror_module
    import github_agent.mcp_server as server_module

    server_source = inspect.getsource(server_module.register_repo_tools)
    mirror_source = inspect.getsource(mirror_module.register_repo_tools)
    assert server_source == mirror_source
    assert (
        server_module.DESTRUCTIVE_REPO_ACTIONS == mirror_module.DESTRUCTIVE_REPO_ACTIONS
    )
