"""Hardening: bounded-by-default pagination + slim payloads + outbound timeout."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from github_agent.api.api_client_base import (
    BaseApiClient,
    _default_timeout,
    _TimeoutAdapter,
)
from github_agent.mcp_server import _slim


def test_default_timeout_env_override(monkeypatch):
    monkeypatch.setenv("GITHUB_HTTP_CONNECT_TIMEOUT", "2")
    monkeypatch.setenv("GITHUB_HTTP_READ_TIMEOUT", "5")
    assert _default_timeout() == (2.0, 5.0)


def test_timeout_adapter_injects_default_only_when_unset(monkeypatch):
    adapter = _TimeoutAdapter((3, 7))
    captured = {}

    import requests

    def fake_parent_send(self, request, **kwargs):
        captured.update(kwargs)
        return requests.Response()

    # Replace the parent send so super().send() in the adapter records the
    # timeout the adapter forwarded, without performing real network I/O.
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_parent_send)

    prepared = requests.Request("GET", "https://api.github.com/x").prepare()

    adapter.send(prepared)  # no timeout -> default injected
    assert captured["timeout"] == (3, 7)
    adapter.send(prepared, timeout=99)  # explicit -> preserved
    assert captured["timeout"] == 99


class _FakeResponse:
    def __init__(self, data, link=None):
        self._data = data
        self.headers = {"Link": link} if link else {}

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _client_without_network() -> BaseApiClient:
    """Build a client without triggering __init__'s /user auth probe."""
    client = object.__new__(BaseApiClient)
    client.url = "https://api.github.com"
    client.headers = {}
    return client


def _model(max_pages=None):
    return SimpleNamespace(api_parameters={"per_page": 30}, max_pages=max_pages)


def test_slim_drops_hypermedia_keeps_html_url_and_data():
    item = {
        "id": 1,
        "name": "ci",
        "html_url": "https://github.com/o/r/runs/1",
        "url": "https://api.github.com/...",
        "jobs_url": "https://api.github.com/.../jobs",
        "_links": {"self": {"href": "x"}},
        "actor": {"login": "octocat", "events_url": "https://api/..."},
    }
    out = _slim([item])[0]
    assert out["id"] == 1 and out["name"] == "ci"
    assert out["html_url"].endswith("/runs/1")  # html_url preserved
    assert "url" not in out and "jobs_url" not in out and "_links" not in out
    assert out["actor"] == {"login": "octocat"}  # nested *_url stripped too


def test_pagination_bounded_to_first_page_by_default(monkeypatch):
    client = _client_without_network()
    client._session = MagicMock()
    client._session.get.return_value = _FakeResponse(
        [{"n": 1}], link='<...?page=5>; rel="last"'
    )
    extra = MagicMock(return_value=[{"n": 99}])  # would fetch pages 2..N
    monkeypatch.setattr(client, "_fetch_next_page", extra)

    _resp, data = client._fetch_all_pages("/x", _model(max_pages=None))

    assert data == [{"n": 1}]  # only page 1
    assert extra.call_count == 0  # no extra pages fetched


def test_max_pages_zero_means_all_pages(monkeypatch):
    client = _client_without_network()
    client._session = MagicMock()
    client._session.get.return_value = _FakeResponse(
        [{"n": 1}], link='<...?page=3>; rel="last"'
    )
    monkeypatch.setattr(
        client, "_fetch_next_page", MagicMock(return_value=[{"n": "more"}])
    )

    _resp, data = client._fetch_all_pages("/x", _model(max_pages=0))

    # page 1 + pages 2 and 3
    assert data[0] == {"n": 1}
    assert data.count({"n": "more"}) == 2
