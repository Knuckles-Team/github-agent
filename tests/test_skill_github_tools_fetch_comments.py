"""Characterization tests for github-tools's fetch_comments.py fetch_all().

fetch_all() pages through a PR's comments/reviews/review-threads via GraphQL
(``gh_api_graphql``, itself a ``gh`` CLI subprocess call). Loaded via importlib
from its file path (no ``__init__.py``; standalone CLI script) and exercised
by monkeypatching ``gh_api_graphql`` — no real ``gh`` invocation.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent
    / "github_agent"
    / "skills"
    / "github-tools"
    / "scripts"
    / "fetch_comments.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_comments", _MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fetch_comments():
    return _load_module()


def _connection(nodes, has_next=False, end_cursor=None):
    return {"nodes": nodes, "pageInfo": {"hasNextPage": has_next, "endCursor": end_cursor}}


def _payload(pr_number=1, comments=None, reviews=None, threads=None):
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "number": pr_number,
                    "url": "https://github.com/o/r/pull/1",
                    "title": "t",
                    "state": "OPEN",
                    "comments": comments or _connection([]),
                    "reviews": reviews or _connection([]),
                    "reviewThreads": threads or _connection([]),
                }
            }
        }
    }


def test_fetch_all_single_page(fetch_comments, monkeypatch):
    payload = _payload(
        comments=_connection([{"id": "c1"}]),
        reviews=_connection([{"id": "rv1"}]),
        threads=_connection([{"id": "t1"}]),
    )
    monkeypatch.setattr(fetch_comments, "gh_api_graphql", lambda **_: payload)

    result = fetch_comments.fetch_all("o", "r", 1)
    assert result["pull_request"] == {
        "number": 1,
        "url": "https://github.com/o/r/pull/1",
        "title": "t",
        "state": "OPEN",
        "owner": "o",
        "repo": "r",
    }
    assert result["conversation_comments"] == [{"id": "c1"}]
    assert result["reviews"] == [{"id": "rv1"}]
    assert result["review_threads"] == [{"id": "t1"}]


def test_fetch_all_paginates_comments_across_multiple_calls(fetch_comments, monkeypatch):
    calls = []
    page1 = _payload(comments=_connection([{"id": "c1"}], has_next=True, end_cursor="CUR1"))
    page2 = _payload(comments=_connection([{"id": "c2"}]))

    def fake_gh_api_graphql(**kwargs):
        calls.append(kwargs)
        return page1 if kwargs.get("comments_cursor") is None else page2

    monkeypatch.setattr(fetch_comments, "gh_api_graphql", fake_gh_api_graphql)

    result = fetch_comments.fetch_all("o", "r", 1)
    assert len(calls) == 2
    assert calls[1]["comments_cursor"] == "CUR1"
    assert result["conversation_comments"] == [{"id": "c1"}, {"id": "c2"}]


def test_fetch_all_raises_on_graphql_errors(fetch_comments, monkeypatch):
    monkeypatch.setattr(
        fetch_comments,
        "gh_api_graphql",
        lambda **_: {"errors": [{"message": "bad query"}]},
    )
    with pytest.raises(RuntimeError, match="GraphQL errors"):
        fetch_comments.fetch_all("o", "r", 1)
