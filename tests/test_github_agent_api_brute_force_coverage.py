import asyncio
import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_session():  # vulture: ignore
    with patch("requests.Session") as mock_s:
        session = mock_s.return_value

        def build_response(url, *args, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.headers = {"Link": '<http://test?page=2>; rel="last"'}

            url_str = str(url)
            if "/search/" in url_str:
                response.json.return_value = {
                    "total_count": 1,
                    "incomplete_results": False,
                    "items": [{"id": 1, "name": "test", "path": "test", "sha": "test"}],
                }
            elif any(x in url_str for x in ["/releases/", "/actions/runs/"]) or (
                "/repos/" in url_str
                and not url_str.endswith(
                    (
                        "/issues",
                        "/pulls",
                        "/commits",
                        "/branches",
                        "/releases",
                        "/keys",
                        "/collaborators",
                        "/members",
                        "/teams",
                        "/repos",
                    )
                )
            ):
                response.json.return_value = {
                    "id": 1,
                    "name": "test",
                    "sha": "abc123sha",
                    "tag_name": "v1.0.0",
                    "head_branch": "main",
                    "head_sha": "abc123sha",
                    "status": "completed",
                    "conclusion": "success",
                }
            else:
                response.json.return_value = [
                    {
                        "id": 1,
                        "name": "test",
                        "sha": "abc123sha",
                        "login": "test",
                        "workflows": [],
                    }
                ]

            response.text = '{"id": 1}'
            return response

        session.get.side_effect = build_response
        session.post.side_effect = build_response
        session.put.side_effect = build_response
        session.delete.side_effect = build_response
        session.patch.side_effect = build_response
        yield session


#: Required params named exactly one of these get a fixed value, checked
#: BEFORE the id-like/annotation/late-name checks below (matches the
#: original if/elif precedence).
_EARLY_NAME_DEFAULTS = {"owner": "test", "repo": "test"}

#: Required params whose name (exactly, not "id" in p_name) means "an id".
_ID_LIKE_NAMES = {"number", "run_id", "release_id"}

#: Checked after the id-like check, before the late name defaults.
_ANNOTATION_DEFAULTS = {int: 1, bool: True, dict: {}, list: []}

#: Checked last, before falling back to the literal string "test".
_LATE_NAME_DEFAULTS = {
    "branch": "main",
    "sha": "abc123sha",
    "ref": "abc123sha",
    "username": "test-user",
    "org": "test-org",
}

#: Extra defaults applied when a method also accepts **kwargs.
_VAR_KEYWORD_DEFAULTS = {
    "q": "test",
    "org": "test-org",
    "path": "test.txt",
    "message": "test-message",
    "content": "test-content",
    "workflows": ["test"],
    "ref": "abc123sha",
    "branch": "main",
    "username": "test-user",
    "state": "open",
    "title": "test-title",
    "head": "main",
    "base": "main",
    "tag_name": "v1.0.0",
}


def _is_id_like(p_name: str) -> bool:
    return "id" in p_name or p_name in _ID_LIKE_NAMES


def _synthesize_required_value(p_name: str, annotation: Any) -> Any:
    """One required parameter's best-effort synthesized value.

    Precedence (matches the original if/elif chain exactly): an early
    owner/repo name, then an id-like name, then the parameter's annotation,
    then a late name default, then the literal fallback "test".
    """
    if p_name in _EARLY_NAME_DEFAULTS:
        return _EARLY_NAME_DEFAULTS[p_name]
    if _is_id_like(p_name):
        return 1
    if annotation in _ANNOTATION_DEFAULTS:
        return _ANNOTATION_DEFAULTS[annotation]
    return _LATE_NAME_DEFAULTS.get(p_name, "test")


def _synthesize_api_call_kwargs(sig: inspect.Signature) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for p_name, p in sig.parameters.items():
        if p.default == inspect.Parameter.empty:
            kwargs[p_name] = _synthesize_required_value(p_name, p.annotation)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        kwargs.update(_VAR_KEYWORD_DEFAULTS)
    return kwargs


@pytest.mark.usefixtures("mock_session")
def test_github_brute_force():
    from github_agent.api_client import Api

    api_instance = Api(url="http://test", token="test")

    # Introspect all methods
    for name, method in inspect.getmembers(api_instance, predicate=inspect.ismethod):
        if name.startswith("_") or name == "authenticate":
            continue

        print(f"Calling {name}...")
        kwargs = _synthesize_api_call_kwargs(inspect.signature(method))

        try:
            method(**kwargs)
        except Exception as e:
            print(f"Operation failed: {type(e).__name__}")


def _synthesize_mcp_tool_params(tool: Any) -> dict[str, str]:
    """Every declared property of ``tool`` gets the placeholder value "test".

    (The original had an `"id" in p or "name" in p` split here whose two
    branches both assigned "test" -- a no-op branch, removed; every property
    still gets exactly the same value.)
    """
    target_params: dict[str, str] = {}
    if hasattr(tool, "parameters") and hasattr(tool.parameters, "properties"):
        for p in tool.parameters.properties:
            target_params[p] = "test"
    return target_params


@pytest.mark.usefixtures("mock_session")
def test_mcp_server_coverage():
    from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

    from github_agent.mcp_server import get_mcp_instance

    async def mock_on_request(self, context, call_next):
        return await call_next(context)

    with patch.object(RateLimitingMiddleware, "on_request", mock_on_request):
        # Mock get_client in mcp_server
        with patch("github_agent.mcp_server.get_client") as mock_gc:
            api = mock_gc.return_value
            api.get_repositories.return_value = MagicMock(data=[])

            mcp_data = get_mcp_instance()
            mcp = mcp_data[0] if isinstance(mcp_data, tuple) else mcp_data

            async def run_tools():
                tool_objs = (
                    await mcp.list_tools()
                    if inspect.iscoroutinefunction(mcp.list_tools)
                    else mcp.list_tools()
                )
                for tool in tool_objs:
                    print(f"Testing MCP tool: {tool.name}")
                    try:
                        target_params = _synthesize_mcp_tool_params(tool)
                        await mcp.call_tool(tool.name, target_params)
                    except Exception as e:
                        print(f"Operation failed: {type(e).__name__}")

            loop = asyncio.new_event_loop()
            loop.run_until_complete(run_tools())
            loop.close()
