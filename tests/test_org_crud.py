"""Organization CRUD coverage: API client methods + the github_orgs MCP tool."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import requests
from agent_utilities.core.exceptions import ParameterError

from github_agent.api.api_client_orgs import OrganizationCreationNotSupportedError
from github_agent.api_client import Api
from github_agent.github_response_models import (
    Organization,
    OrganizationMembership,
    OrganizationSummary,
    Repository,
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

ORG_PAYLOAD = {
    "login": "Knuckles-Team",
    "id": 1,
    "node_id": "O_1",
    "url": "https://api.github.com/orgs/Knuckles-Team",
    "description": "Test org",
    "name": "Knuckles Team",
    "billing_email": "billing@example.com",
    "company": "Knuckles",
    "default_repository_permission": "read",
    "members_can_create_repositories": True,
    "web_commit_signoff_required": False,
}

REPO_PAYLOAD = {
    "id": 99,
    "node_id": "R_99",
    "name": "service",
    "full_name": "Knuckles-Team/service",
    "private": True,
    "owner": USER_PAYLOAD,
    "html_url": "https://github.com/Knuckles-Team/service",
    "description": "d",
    "fork": False,
    "url": "https://api.github.com/repos/Knuckles-Team/service",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "pushed_at": "2026-01-01T00:00:00Z",
    "git_url": "git://github.com/Knuckles-Team/service.git",
    "ssh_url": "git@github.com:Knuckles-Team/service.git",
    "clone_url": "https://github.com/Knuckles-Team/service.git",
    "svn_url": "https://github.com/Knuckles-Team/service",
    "size": 1,
    "stargazers_count": 0,
    "watchers_count": 0,
    "has_issues": True,
    "has_projects": True,
    "has_downloads": True,
    "has_wiki": True,
    "has_pages": False,
    "forks_count": 0,
    "archived": False,
    "disabled": False,
    "open_issues_count": 0,
    "allow_forking": True,
    "is_template": False,
    "topics": [],
    "visibility": "private",
    "forks": 0,
    "open_issues": 0,
    "watchers": 0,
    "default_branch": "main",
}

MEMBERSHIP_PAYLOAD = {
    "url": "https://api.github.com/orgs/Knuckles-Team/memberships/octocat",
    "state": "active",
    "role": "admin",
    "organization_url": "https://api.github.com/orgs/Knuckles-Team",
    "organization": {"login": "Knuckles-Team", "id": 1},
    "user": USER_PAYLOAD,
}


def make_response(status_code=200, json_data=None, headers=None):
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = json_data if json_data is not None else {}
    response.text = "{}"
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


def test_get_organization(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=ORG_PAYLOAD)

    res = api.get_organization(org="Knuckles-Team")

    assert isinstance(res.data, Organization)
    assert res.data.login == "Knuckles-Team"
    assert res.data.billing_email == "billing@example.com"
    mock_session.get.assert_called_with(
        url="https://api.github.com/orgs/Knuckles-Team",
        headers=api.headers,
    )


def test_list_organizations_member_scope(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(
        json_data=[{"login": "Knuckles-Team", "id": 1}]
    )

    res = api.list_organizations()

    assert len(res.data) == 1
    assert isinstance(res.data[0], OrganizationSummary)
    call = mock_session.get.call_args
    assert call.kwargs["url"] == "https://api.github.com/user/orgs"


def test_list_organizations_all_scope_since_cursor(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(
        json_data=[{"login": "after-cursor", "id": 43}]
    )

    res = api.list_organizations(scope="all", since=42)

    assert res.data[0].id == 43
    call = mock_session.get.call_args
    assert call.kwargs["url"] == "https://api.github.com/organizations"
    assert call.kwargs["params"]["since"] == 42


def test_update_organization_field_passthrough(mock_session):
    api = Api(token="test")
    mock_session.patch.return_value = make_response(json_data=ORG_PAYLOAD)

    res = api.update_organization(
        org="Knuckles-Team",
        billing_email="billing@example.com",
        company="Knuckles",
        web_commit_signoff_required=False,
        default_repository_permission="read",
    )

    assert isinstance(res.data, Organization)
    mock_session.patch.assert_called_with(
        url="https://api.github.com/orgs/Knuckles-Team",
        json={
            "billing_email": "billing@example.com",
            "company": "Knuckles",
            "default_repository_permission": "read",
            "web_commit_signoff_required": False,
        },
        headers=api.headers,
    )


def test_update_organization_rejects_unknown_field(mock_session):
    api = Api(token="test")
    with pytest.raises(ParameterError):
        api.update_organization(org="Knuckles-Team", not_a_real_field="x")
    mock_session.patch.assert_not_called()


def test_delete_organization_202(mock_session):
    api = Api(token="test")
    mock_session.delete.return_value = make_response(status_code=202)

    res = api.delete_organization(org="Knuckles-Team")

    assert res.response.status_code == 202
    assert res.data == {"status": "deletion_scheduled"}
    mock_session.delete.assert_called_with(
        url="https://api.github.com/orgs/Knuckles-Team",
        headers=api.headers,
    )


def test_create_organization_rejected_on_github_com(mock_session):
    api = Api(token="test")  # default url: https://api.github.com

    with pytest.raises(OrganizationCreationNotSupportedError) as exc:
        api.create_organization(login="neworg", admin="octocat")

    assert "web UI" in str(exc.value)
    assert "Enterprise Server" in str(exc.value)
    mock_session.post.assert_not_called()


def test_create_organization_on_enterprise_server(mock_session):
    api = Api(url="https://ghe.example.com/api/v3", token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data={"login": "neworg", "id": 5}
    )

    res = api.create_organization(
        login="neworg", admin="octocat", profile_name="New Org"
    )

    assert isinstance(res.data, OrganizationSummary)
    assert res.data.login == "neworg"
    mock_session.post.assert_called_with(
        url="https://ghe.example.com/api/v3/admin/organizations",
        json={"login": "neworg", "admin": "octocat", "profile_name": "New Org"},
        headers=api.headers,
    )


def test_org_repo_create_payload_parity_with_user_repo_create(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data=REPO_PAYLOAD
    )
    repo_kwargs = {"private": True, "description": "d", "auto_init": True}

    user_res = api.create_repository(name="service", **repo_kwargs)
    org_res = api.create_organization_repository(
        org="Knuckles-Team", name="service", **repo_kwargs
    )

    assert isinstance(user_res.data, Repository)
    assert isinstance(org_res.data, Repository)
    user_call, org_call = mock_session.post.call_args_list
    assert user_call.kwargs["json"] == org_call.kwargs["json"]
    assert user_call.kwargs["url"] == "https://api.github.com/user/repos"
    assert org_call.kwargs["url"] == "https://api.github.com/orgs/Knuckles-Team/repos"


def test_get_organization_membership(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=MEMBERSHIP_PAYLOAD)

    res = api.get_organization_membership(org="Knuckles-Team", username="octocat")

    assert isinstance(res.data, OrganizationMembership)
    assert res.data.role == "admin"
    assert res.data.state == "active"
    call = mock_session.get.call_args
    assert (
        call.kwargs["url"]
        == "https://api.github.com/orgs/Knuckles-Team/memberships/octocat"
    )


def test_set_organization_membership(mock_session):
    api = Api(token="test")
    mock_session.put.return_value = make_response(json_data=MEMBERSHIP_PAYLOAD)

    res = api.set_organization_membership(
        org="Knuckles-Team", username="octocat", role="admin"
    )

    assert isinstance(res.data, OrganizationMembership)
    mock_session.put.assert_called_with(
        url="https://api.github.com/orgs/Knuckles-Team/memberships/octocat",
        json={"role": "admin"},
        headers=api.headers,
    )


def test_remove_organization_member(mock_session):
    api = Api(token="test")
    mock_session.delete.return_value = make_response(status_code=204)

    res = api.remove_organization_member(org="Knuckles-Team", username="octocat")

    assert res.data == {"status": "removed"}
    mock_session.delete.assert_called_with(
        url="https://api.github.com/orgs/Knuckles-Team/members/octocat",
        headers=api.headers,
    )


# --- MCP tool ----------------------------------------------------------------


class AsyncMockContext:
    def __init__(self):
        self.info_calls = []

    async def info(self, msg):
        self.info_calls.append(msg)


def make_org_client():
    client = MagicMock()
    mock_org = MagicMock()
    mock_org.model_dump.return_value = {"login": "Knuckles-Team", "id": 1}
    client.get_organization.return_value = MagicMock(data=mock_org)
    client.list_organizations.return_value = MagicMock(data=[mock_org])
    client.update_organization.return_value = MagicMock(data=mock_org)
    client.delete_organization.return_value = MagicMock(
        data={"status": "deletion_scheduled"}
    )
    client.create_organization.return_value = MagicMock(data=mock_org)
    mock_repo = MagicMock()
    mock_repo.model_dump.return_value = {"id": 99, "name": "service"}
    client.create_organization_repository.return_value = MagicMock(data=mock_repo)
    mock_membership = MagicMock()
    mock_membership.model_dump.return_value = {"state": "active", "role": "member"}
    client.get_organization_membership.return_value = MagicMock(data=mock_membership)
    client.set_organization_membership.return_value = MagicMock(data=mock_membership)
    client.remove_organization_member.return_value = MagicMock(
        data={"status": "removed"}
    )
    return client


async def get_github_orgs_tool():
    from github_agent.mcp_server import get_mcp_instance

    mcp = get_mcp_instance()[0]
    if inspect.iscoroutinefunction(mcp.list_tools):
        tools = await mcp.list_tools()
    else:
        tools = mcp.list_tools()
    return {t.name: t.fn for t in tools}["github_orgs"]


@pytest.mark.anyio
async def test_mcp_orgs_get_list_update():
    github_orgs = await get_github_orgs_tool()
    client = make_org_client()
    ctx = AsyncMockContext()

    res = await github_orgs(
        action="get", params_json='{"org": "o"}', client=client, ctx=ctx
    )
    assert res["status"] == 200
    assert res["data"]["login"] == "Knuckles-Team"

    res = await github_orgs(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_orgs(
        action="list",
        params_json='{"scope": "all", "since": 42}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.list_organizations.assert_called_with(scope="all", since=42)

    res = await github_orgs(
        action="update",
        params_json='{"org": "o", "billing_email": "b@x.com"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.update_organization.assert_called_with(org="o", billing_email="b@x.com")

    res = await github_orgs(action="update", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_orgs_delete_destructive_gating(monkeypatch):
    monkeypatch.delenv("GITHUB_ALLOW_DESTRUCTIVE", raising=False)
    github_orgs = await get_github_orgs_tool()
    client = make_org_client()
    ctx = AsyncMockContext()

    # Blocked by default
    res = await github_orgs(
        action="delete",
        params_json='{"org": "o"}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403
    assert "allow_destructive" in res["error"]
    client.delete_organization.assert_not_called()

    # Allowed with explicit per-call consent
    res = await github_orgs(
        action="delete",
        params_json='{"org": "o"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 202
    assert res["data"] == {"status": "deletion_scheduled"}

    # Allowed via the environment default
    monkeypatch.setenv("GITHUB_ALLOW_DESTRUCTIVE", "True")
    res = await github_orgs(
        action="delete",
        params_json='{"org": "o"}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 202

    # Missing org still 400 (with gate open)
    res = await github_orgs(
        action="delete",
        params_json="{}",
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_orgs_create_github_com_limitation():
    github_orgs = await get_github_orgs_tool()
    client = make_org_client()
    ctx = AsyncMockContext()

    client.create_organization.side_effect = OrganizationCreationNotSupportedError(
        "github.com organizations cannot be created via the API — web UI only."
    )
    res = await github_orgs(
        action="create",
        params_json='{"login": "neworg", "admin": "octocat"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400
    assert "web UI" in res["error"]

    # Enterprise success path
    client.create_organization.side_effect = None
    res = await github_orgs(
        action="create",
        params_json='{"login": "neworg", "admin": "octocat", "profile_name": "N"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    client.create_organization.assert_called_with(
        login="neworg", admin="octocat", profile_name="N"
    )

    res = await github_orgs(
        action="create", params_json='{"login": "x"}', client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_orgs_create_repository():
    github_orgs = await get_github_orgs_tool()
    client = make_org_client()
    ctx = AsyncMockContext()

    res = await github_orgs(
        action="create_repository",
        params_json='{"org": "o", "name": "service", "private": true}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    client.create_organization_repository.assert_called_with(
        org="o", name="service", private=True
    )

    res = await github_orgs(
        action="create_repository", params_json='{"org": "o"}', client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_orgs_membership_actions(monkeypatch):
    monkeypatch.delenv("GITHUB_ALLOW_DESTRUCTIVE", raising=False)
    github_orgs = await get_github_orgs_tool()
    client = make_org_client()
    ctx = AsyncMockContext()

    res = await github_orgs(
        action="get_membership",
        params_json='{"org": "o", "username": "u"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_orgs(
        action="get_membership", params_json='{"org": "o"}', client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_orgs(
        action="set_membership",
        params_json='{"org": "o", "username": "u", "role": "admin"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.set_organization_membership.assert_called_with(
        org="o", username="u", role="admin"
    )

    # remove_member is destructive: blocked by default, allowed with consent
    res = await github_orgs(
        action="remove_member",
        params_json='{"org": "o", "username": "u"}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403
    client.remove_organization_member.assert_not_called()

    res = await github_orgs(
        action="remove_member",
        params_json='{"org": "o", "username": "u"}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"] == {"status": "removed"}


@pytest.mark.anyio
async def test_mcp_orgs_module_mirror_registers_same_actions():
    """The mcp package mirror must expose the same org tool as mcp_server."""
    from fastmcp import FastMCP

    from github_agent.mcp.mcp_org import register_org_tools

    mcp = FastMCP("mirror-check")
    register_org_tools(mcp)
    if inspect.iscoroutinefunction(mcp.list_tools):
        tools = await mcp.list_tools()
    else:
        tools = mcp.list_tools()
    tool = {t.name: t for t in tools}["github_orgs"]
    description = tool.description or ""
    for action in (
        "get",
        "list",
        "update",
        "delete",
        "create",
        "create_repository",
        "get_membership",
        "set_membership",
        "remove_member",
        "teams",
    ):
        assert f"'{action}'" in description
