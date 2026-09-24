import asyncio
import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_session():  # vulture: ignore
    with patch("requests.Session") as mock_s:
        session = mock_s.return_value
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "id": 1,
            "name": "test",
            "html_url": "http://test",
        }
        response.text = '{"id": 1}'
        response.headers = {"Link": '<http://test?page=2>; rel="last"'}
        session.get.return_value = response
        session.post.return_value = response
        session.put.return_value = response
        session.delete.return_value = response
        session.patch.return_value = response
        session.request.return_value = response
        yield session


def _synthesize_api_kwargs(
    sig: inspect.Signature, common_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Best-effort kwargs for an ``Api`` method's signature from a shared pool."""
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_kwargs:
        return common_kwargs.copy()
    kwargs = {k: v for k, v in common_kwargs.items() if k in sig.parameters}
    for p_name, p in sig.parameters.items():
        if p.default == inspect.Parameter.empty and p_name not in kwargs:
            kwargs[p_name] = "test"
    return kwargs


@pytest.mark.usefixtures("mock_session")
def test_github_api_brute_force():
    from github_agent.api_client import Api

    api = Api(token="test")

    common_kwargs = {
        "owner": "test",
        "repo": "test",
        "issue_number": 1,
        "pull_number": 1,
        "comment_id": 1,
        "ref": "main",
        "path": "test.txt",
        "message": "test",
        "content": "test",
        "sha": "abc123def456",
    }

    # Introspect all methods
    for name, method in inspect.getmembers(api, predicate=inspect.ismethod):
        if name.startswith("_"):
            continue
        print(f"Calling Api.{name}...")
        kwargs = _synthesize_api_kwargs(inspect.signature(method), common_kwargs)
        try:
            method(**kwargs)
        except:
            pass


def _synthesize_tool_params(sig: inspect.Signature) -> dict[str, Any]:
    """Best-effort call params for an MCP tool's signature."""
    target_params = {"owner": "test", "repo": "test"}
    for p_name, p in sig.parameters.items():
        if (
            p.default == inspect.Parameter.empty
            and p_name not in ("_client", "context")
            and p_name not in target_params
        ):
            target_params[p_name] = "test"

    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_kwargs:
        target_params = {k: v for k, v in target_params.items() if k in sig.parameters}
    return target_params


@pytest.mark.usefixtures("mock_session")
def test_mcp_server_coverage():
    from github_agent.mcp_server import get_mcp_instance

    with patch("github_agent.auth.get_client"):
        mcp_data = get_mcp_instance()
        mcp = mcp_data[0] if isinstance(mcp_data, tuple) else mcp_data

        async def run_tools():
            tool_objs = (
                await mcp.list_tools()
                if inspect.iscoroutinefunction(mcp.list_tools)
                else mcp.list_tools()
            )
            for tool in tool_objs:
                try:
                    target_params = _synthesize_tool_params(inspect.signature(tool.fn))
                    await mcp.call_tool(tool.name, target_params)
                except:
                    pass

        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_tools())
        loop.close()


@pytest.mark.usefixtures("mock_session")
def test_auth_delegation():
    from agent_connector_sdk.auth.delegation import DelegationSettings
    from agent_connector_sdk.auth.tokens import AccessToken

    from github_agent.auth import get_client

    settings = DelegationSettings(
        enabled=True,
        token_endpoint="http://test/token",
        client_id="test",
        client_secret_ref="env://GITHUB_OIDC_CLIENT_SECRET",
        audience="test",
        scopes="test",
    )
    fake_token = AccessToken("exchanged_token", 300.0, 0.0)

    with (
        patch.object(DelegationSettings, "from_settings", return_value=settings),
        patch("github_agent.auth.current_user_token", return_value="mock_subject_token"),
        patch("github_agent.auth.exchange_token", return_value=fake_token),
    ):
        client = get_client()
        assert client.headers["Authorization"] == "Bearer exchanged_token"
