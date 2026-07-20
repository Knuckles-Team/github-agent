"""Comment CRUD coverage: API client methods + the github_comments MCP tool.

Covers issue/PR comments (a pull request IS an issue on GitHub, so the
issue-comment endpoints serve both), pull-request review (inline code)
comments, and commit comments.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import requests

from github_agent.api_client import Api


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
        session.patch.return_value = default
        session.delete.return_value = default
        yield session


# --- API client: issue/PR comments -------------------------------------


def test_list_issue_comments(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[{"id": 1, "body": "hi"}])

    res = api.list_issue_comments(owner="o", repo="r", issue_number=5)

    assert res.data == [{"id": 1, "body": "hi"}]
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/issues/5/comments",
        params=None,
        headers=api.headers,
    )


def test_list_issue_comments_passes_filters(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[])

    api.list_issue_comments(
        owner="o", repo="r", issue_number=5, since="2026-01-01T00:00:00Z", per_page=10
    )

    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/issues/5/comments",
        params={"since": "2026-01-01T00:00:00Z", "per_page": 10},
        headers=api.headers,
    )


def test_list_repo_issue_comments(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[{"id": 2}])

    res = api.list_repo_issue_comments(owner="o", repo="r", sort="updated")

    assert res.data == [{"id": 2}]
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/issues/comments",
        params={"sort": "updated"},
        headers=api.headers,
    )


def test_get_issue_comment(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data={"id": 3, "body": "hello"})

    res = api.get_issue_comment(owner="o", repo="r", comment_id=3)

    assert res.data == {"id": 3, "body": "hello"}
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/issues/comments/3",
        headers=api.headers,
    )


def test_create_issue_comment(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data={"id": 4, "body": "new comment"}
    )

    res = api.create_issue_comment(
        owner="o", repo="r", issue_number=5, body="new comment"
    )

    assert res.data == {"id": 4, "body": "new comment"}
    mock_session.post.assert_called_with(
        url="https://api.github.com/repos/o/r/issues/5/comments",
        json={"body": "new comment"},
        headers=api.headers,
    )


def test_update_issue_comment(mock_session):
    api = Api(token="test")
    mock_session.patch.return_value = make_response(
        json_data={"id": 4, "body": "edited"}
    )

    res = api.update_issue_comment(owner="o", repo="r", comment_id=4, body="edited")

    assert res.data == {"id": 4, "body": "edited"}
    mock_session.patch.assert_called_with(
        url="https://api.github.com/repos/o/r/issues/comments/4",
        json={"body": "edited"},
        headers=api.headers,
    )


def test_delete_issue_comment(mock_session):
    api = Api(token="test")
    mock_session.delete.return_value = make_response(status_code=204)

    res = api.delete_issue_comment(owner="o", repo="r", comment_id=4)

    assert res.data == {"status": "deleted"}
    mock_session.delete.assert_called_with(
        url="https://api.github.com/repos/o/r/issues/comments/4",
        headers=api.headers,
    )


# --- API client: pull-request review (inline) comments -----------------


def test_list_review_comments(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[{"id": 10}])

    res = api.list_review_comments(owner="o", repo="r", pull_number=7)

    assert res.data == [{"id": 10}]
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/pulls/7/comments",
        params=None,
        headers=api.headers,
    )


def test_list_repo_review_comments(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[{"id": 11}])

    res = api.list_repo_review_comments(owner="o", repo="r", direction="desc")

    assert res.data == [{"id": 11}]
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/pulls/comments",
        params={"direction": "desc"},
        headers=api.headers,
    )


def test_get_review_comment(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(
        json_data={"id": 12, "body": "nice catch"}
    )

    res = api.get_review_comment(owner="o", repo="r", comment_id=12)

    assert res.data == {"id": 12, "body": "nice catch"}
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/pulls/comments/12",
        headers=api.headers,
    )


def test_create_review_comment(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data={"id": 13}
    )

    res = api.create_review_comment(
        owner="o",
        repo="r",
        pull_number=7,
        body="inline comment",
        commit_id="abc123",
        path="file.py",
        line=10,
        side="RIGHT",
    )

    assert res.data == {"id": 13}
    mock_session.post.assert_called_with(
        url="https://api.github.com/repos/o/r/pulls/7/comments",
        json={
            "body": "inline comment",
            "commit_id": "abc123",
            "path": "file.py",
            "line": 10,
            "side": "RIGHT",
        },
        headers=api.headers,
    )


def test_create_review_comment_reply(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data={"id": 14}
    )

    res = api.create_review_comment_reply(
        owner="o", repo="r", pull_number=7, comment_id=13, body="agreed"
    )

    assert res.data == {"id": 14}
    mock_session.post.assert_called_with(
        url="https://api.github.com/repos/o/r/pulls/7/comments/13/replies",
        json={"body": "agreed"},
        headers=api.headers,
    )


def test_update_review_comment(mock_session):
    api = Api(token="test")
    mock_session.patch.return_value = make_response(
        json_data={"id": 13, "body": "edited inline"}
    )

    res = api.update_review_comment(
        owner="o", repo="r", comment_id=13, body="edited inline"
    )

    assert res.data == {"id": 13, "body": "edited inline"}
    mock_session.patch.assert_called_with(
        url="https://api.github.com/repos/o/r/pulls/comments/13",
        json={"body": "edited inline"},
        headers=api.headers,
    )


def test_delete_review_comment(mock_session):
    api = Api(token="test")
    mock_session.delete.return_value = make_response(status_code=204)

    res = api.delete_review_comment(owner="o", repo="r", comment_id=13)

    assert res.data == {"status": "deleted"}
    mock_session.delete.assert_called_with(
        url="https://api.github.com/repos/o/r/pulls/comments/13",
        headers=api.headers,
    )


# --- API client: commit comments ----------------------------------------


def test_list_commit_comments(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[{"id": 20}])

    res = api.list_commit_comments(owner="o", repo="r", sha="abc123")

    assert res.data == [{"id": 20}]
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/commits/abc123/comments",
        params=None,
        headers=api.headers,
    )


def test_list_commit_comments_passes_filters(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[])

    api.list_commit_comments(owner="o", repo="r", sha="abc123", per_page=10)

    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/commits/abc123/comments",
        params={"per_page": 10},
        headers=api.headers,
    )


def test_list_repo_commit_comments(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(json_data=[{"id": 21}])

    res = api.list_repo_commit_comments(owner="o", repo="r", per_page=5)

    assert res.data == [{"id": 21}]
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/comments",
        params={"per_page": 5},
        headers=api.headers,
    )


def test_get_commit_comment(mock_session):
    api = Api(token="test")
    mock_session.get.return_value = make_response(
        json_data={"id": 22, "body": "nice commit"}
    )

    res = api.get_commit_comment(owner="o", repo="r", comment_id=22)

    assert res.data == {"id": 22, "body": "nice commit"}
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/o/r/comments/22",
        headers=api.headers,
    )


def test_create_commit_comment(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data={"id": 23}
    )

    res = api.create_commit_comment(
        owner="o",
        repo="r",
        sha="abc123",
        body="commit comment",
        path="file.py",
        line=10,
    )

    assert res.data == {"id": 23}
    mock_session.post.assert_called_with(
        url="https://api.github.com/repos/o/r/commits/abc123/comments",
        json={"body": "commit comment", "path": "file.py", "line": 10},
        headers=api.headers,
    )


def test_create_commit_comment_body_only(mock_session):
    api = Api(token="test")
    mock_session.post.return_value = make_response(
        status_code=201, json_data={"id": 24}
    )

    api.create_commit_comment(owner="o", repo="r", sha="abc123", body="plain")

    mock_session.post.assert_called_with(
        url="https://api.github.com/repos/o/r/commits/abc123/comments",
        json={"body": "plain"},
        headers=api.headers,
    )


def test_update_commit_comment(mock_session):
    api = Api(token="test")
    mock_session.patch.return_value = make_response(
        json_data={"id": 23, "body": "edited commit comment"}
    )

    res = api.update_commit_comment(
        owner="o", repo="r", comment_id=23, body="edited commit comment"
    )

    assert res.data == {"id": 23, "body": "edited commit comment"}
    mock_session.patch.assert_called_with(
        url="https://api.github.com/repos/o/r/comments/23",
        json={"body": "edited commit comment"},
        headers=api.headers,
    )


def test_delete_commit_comment(mock_session):
    api = Api(token="test")
    mock_session.delete.return_value = make_response(status_code=204)

    res = api.delete_commit_comment(owner="o", repo="r", comment_id=23)

    assert res.data == {"status": "deleted"}
    mock_session.delete.assert_called_with(
        url="https://api.github.com/repos/o/r/comments/23",
        headers=api.headers,
    )


# --- MCP tool ------------------------------------------------------------


class AsyncMockContext:
    def __init__(self):
        self.info_calls = []

    async def info(self, msg):
        self.info_calls.append(msg)


def make_comment_client():
    client = MagicMock()
    client.list_issue_comments.return_value = MagicMock(data=[{"id": 1, "body": "hi"}])
    client.list_repo_issue_comments.return_value = MagicMock(
        data=[{"id": 1, "body": "hi"}]
    )
    client.get_issue_comment.return_value = MagicMock(data={"id": 1, "body": "hi"})
    client.create_issue_comment.return_value = MagicMock(data={"id": 2, "body": "new"})
    client.update_issue_comment.return_value = MagicMock(
        data={"id": 2, "body": "edited"}
    )
    client.delete_issue_comment.return_value = MagicMock(data={"status": "deleted"})
    client.list_review_comments.return_value = MagicMock(data=[{"id": 10}])
    client.list_repo_review_comments.return_value = MagicMock(data=[{"id": 10}])
    client.get_review_comment.return_value = MagicMock(data={"id": 10})
    client.create_review_comment.return_value = MagicMock(data={"id": 11})
    client.create_review_comment_reply.return_value = MagicMock(data={"id": 12})
    client.update_review_comment.return_value = MagicMock(
        data={"id": 11, "body": "edited"}
    )
    client.delete_review_comment.return_value = MagicMock(data={"status": "deleted"})
    client.list_commit_comments.return_value = MagicMock(data=[{"id": 20}])
    client.list_repo_commit_comments.return_value = MagicMock(data=[{"id": 20}])
    client.get_commit_comment.return_value = MagicMock(data={"id": 20})
    client.create_commit_comment.return_value = MagicMock(data={"id": 21})
    client.update_commit_comment.return_value = MagicMock(
        data={"id": 20, "body": "edited"}
    )
    client.delete_commit_comment.return_value = MagicMock(data={"status": "deleted"})
    return client


async def get_github_comments_tool():
    from github_agent.mcp_server import get_mcp_instance

    mcp = get_mcp_instance()[0]
    if inspect.iscoroutinefunction(mcp.list_tools):
        tools = await mcp.list_tools()
    else:
        tools = mcp.list_tools()
    return {t.name: t.fn for t in tools}["github_comments"]


@pytest.mark.anyio
async def test_mcp_comments_registers():
    """The github_comments tool is registered and reachable through the live server."""
    github_comments = await get_github_comments_tool()
    assert callable(github_comments)


@pytest.mark.anyio
async def test_mcp_comments_issue_crud():
    github_comments = await get_github_comments_tool()
    client = make_comment_client()
    ctx = AsyncMockContext()

    res = await github_comments(
        action="list",
        params_json='{"owner": "o", "repo": "r", "issue_number": 5}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"] == [{"id": 1, "body": "hi"}]
    client.list_issue_comments.assert_called_with(owner="o", repo="r", issue_number=5)

    res = await github_comments(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_comments(
        action="get",
        params_json='{"owner": "o", "repo": "r", "comment_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"] == {"id": 1, "body": "hi"}

    res = await github_comments(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_comments(
        action="create",
        params_json='{"owner": "o", "repo": "r", "issue_number": 5, "body": "new"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    client.create_issue_comment.assert_called_with(
        owner="o", repo="r", issue_number=5, body="new"
    )

    res = await github_comments(
        action="create", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="update",
        params_json='{"owner": "o", "repo": "r", "comment_id": 2, "body": "edited"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.update_issue_comment.assert_called_with(
        owner="o", repo="r", comment_id=2, body="edited"
    )

    res = await github_comments(
        action="update", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    with pytest.raises(ValueError, match="list_actions"):
        await github_comments(
            action="invalid", params_json="{}", client=client, ctx=ctx
        )


@pytest.mark.anyio
async def test_mcp_comments_list_repo():
    github_comments = await get_github_comments_tool()
    client = make_comment_client()
    ctx = AsyncMockContext()

    res = await github_comments(
        action="list_repo",
        params_json='{"owner": "o", "repo": "r", "sort": "updated"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.list_repo_issue_comments.assert_called_with(
        owner="o", repo="r", sort="updated"
    )

    res = await github_comments(
        action="list_repo", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_comments_review_crud():
    github_comments = await get_github_comments_tool()
    client = make_comment_client()
    ctx = AsyncMockContext()

    res = await github_comments(
        action="list_review",
        params_json='{"owner": "o", "repo": "r", "pull_number": 7}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.list_review_comments.assert_called_with(owner="o", repo="r", pull_number=7)

    res = await github_comments(
        action="list_review", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="list_repo_review",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.list_repo_review_comments.assert_called_with(owner="o", repo="r")

    res = await github_comments(
        action="list_repo_review", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="get_review",
        params_json='{"owner": "o", "repo": "r", "comment_id": 10}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_comments(
        action="get_review", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="create_review",
        params_json=(
            '{"owner": "o", "repo": "r", "pull_number": 7, "body": "b", '
            '"commit_id": "abc", "path": "f.py", "line": 3, "side": "RIGHT"}'
        ),
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    client.create_review_comment.assert_called_with(
        owner="o",
        repo="r",
        pull_number=7,
        body="b",
        commit_id="abc",
        path="f.py",
        line=3,
        side="RIGHT",
    )

    res = await github_comments(
        action="create_review", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="reply_review",
        params_json='{"owner": "o", "repo": "r", "pull_number": 7, "comment_id": 10, "body": "b"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    client.create_review_comment_reply.assert_called_with(
        owner="o", repo="r", pull_number=7, comment_id=10, body="b"
    )

    res = await github_comments(
        action="reply_review", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="update_review",
        params_json='{"owner": "o", "repo": "r", "comment_id": 10, "body": "edited"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.update_review_comment.assert_called_with(
        owner="o", repo="r", comment_id=10, body="edited"
    )

    res = await github_comments(
        action="update_review", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_comments_commit_crud():
    github_comments = await get_github_comments_tool()
    client = make_comment_client()
    ctx = AsyncMockContext()

    res = await github_comments(
        action="list_commit",
        params_json='{"owner": "o", "repo": "r", "sha": "abc123"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"] == [{"id": 20}]
    client.list_commit_comments.assert_called_with(owner="o", repo="r", sha="abc123")

    res = await github_comments(
        action="list_commit", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="list_repo_commit",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.list_repo_commit_comments.assert_called_with(owner="o", repo="r")

    res = await github_comments(
        action="list_repo_commit", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="get_commit",
        params_json='{"owner": "o", "repo": "r", "comment_id": 20}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    assert res["data"] == {"id": 20}
    client.get_commit_comment.assert_called_with(owner="o", repo="r", comment_id=20)

    res = await github_comments(
        action="get_commit", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="create_commit",
        params_json=(
            '{"owner": "o", "repo": "r", "sha": "abc123", "body": "b", '
            '"path": "f.py", "line": 3}'
        ),
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201
    client.create_commit_comment.assert_called_with(
        owner="o", repo="r", sha="abc123", body="b", path="f.py", line=3
    )

    res = await github_comments(
        action="create_commit", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_comments(
        action="update_commit",
        params_json='{"owner": "o", "repo": "r", "comment_id": 20, "body": "edited"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.update_commit_comment.assert_called_with(
        owner="o", repo="r", comment_id=20, body="edited"
    )

    res = await github_comments(
        action="update_commit", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_comments_destructive_gating(monkeypatch):
    monkeypatch.delenv("GITHUB_ALLOW_DESTRUCTIVE", raising=False)
    github_comments = await get_github_comments_tool()
    client = make_comment_client()
    ctx = AsyncMockContext()

    # 'delete' blocked by default
    res = await github_comments(
        action="delete",
        params_json='{"owner": "o", "repo": "r", "comment_id": 2}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403
    assert "allow_destructive" in res["error"]
    client.delete_issue_comment.assert_not_called()

    # Allowed with explicit per-call consent
    res = await github_comments(
        action="delete",
        params_json='{"owner": "o", "repo": "r", "comment_id": 2}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.delete_issue_comment.assert_called_with(owner="o", repo="r", comment_id=2)

    # 'delete_review' blocked by default
    res = await github_comments(
        action="delete_review",
        params_json='{"owner": "o", "repo": "r", "comment_id": 10}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403
    client.delete_review_comment.assert_not_called()

    # Allowed via the environment default
    monkeypatch.setenv("GITHUB_ALLOW_DESTRUCTIVE", "True")
    res = await github_comments(
        action="delete_review",
        params_json='{"owner": "o", "repo": "r", "comment_id": 10}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.delete_review_comment.assert_called_with(owner="o", repo="r", comment_id=10)

    # 'delete_commit' blocked by default
    monkeypatch.delenv("GITHUB_ALLOW_DESTRUCTIVE", raising=False)
    res = await github_comments(
        action="delete_commit",
        params_json='{"owner": "o", "repo": "r", "comment_id": 20}',
        allow_destructive=False,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 403
    client.delete_commit_comment.assert_not_called()

    # Allowed with explicit per-call consent
    res = await github_comments(
        action="delete_commit",
        params_json='{"owner": "o", "repo": "r", "comment_id": 20}',
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200
    client.delete_commit_comment.assert_called_with(owner="o", repo="r", comment_id=20)

    # Missing comment_id still 400 (with gate open)
    res = await github_comments(
        action="delete",
        params_json="{}",
        allow_destructive=True,
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_comments_list_actions_discovery():
    github_comments = await get_github_comments_tool()
    client = make_comment_client()
    ctx = AsyncMockContext()

    res = await github_comments(
        action="list_actions", params_json="{}", client=client, ctx=ctx
    )
    assert res["service"] == "github-agent"
    for action in (
        "list",
        "list_repo",
        "get",
        "create",
        "update",
        "delete",
        "list_review",
        "list_repo_review",
        "get_review",
        "create_review",
        "reply_review",
        "update_review",
        "delete_review",
        "list_commit",
        "list_repo_commit",
        "get_commit",
        "create_commit",
        "update_commit",
        "delete_commit",
    ):
        assert action in res["actions"]


@pytest.mark.anyio
async def test_mcp_comments_exception_handling():
    github_comments = await get_github_comments_tool()
    client = make_comment_client()
    ctx = AsyncMockContext()

    client.get_issue_comment.side_effect = Exception("boom")
    res = await github_comments(
        action="get",
        params_json='{"owner": "o", "repo": "r", "comment_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 500


@pytest.mark.anyio
async def test_mcp_comments_invalid_json():
    github_comments = await get_github_comments_tool()
    client = make_comment_client()
    ctx = AsyncMockContext()

    res = await github_comments(
        action="list", params_json="not json", client=client, ctx=ctx
    )
    assert res["status"] == 400
