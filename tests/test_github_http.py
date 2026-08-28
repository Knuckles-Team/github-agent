"""Characterization tests for the fixed-origin GitHub HTTPS client.

`github_agent.github_http.github_json_request` is a small, dependency-free
request helper used by standalone skill scripts. It enforces several
security-relevant bounds directly (method allowlist, path shape, token
shape, request/response size caps, redirect rejection) rather than relying
on `requests`/`Api`, so these tests pin each guard individually before any
refactor of the function.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from github_agent.github_http import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    github_json_request,
)


def _mock_connection(status=200, body=b'{"ok":true}', content_length=None):
    """Build a MagicMock standing in for http.client.HTTPSConnection."""
    response = MagicMock()
    response.status = status
    response.read.return_value = body
    response.getheader.return_value = (
        content_length if content_length is not None else str(len(body))
    )
    connection = MagicMock()
    connection.getresponse.return_value = response
    return connection, response


def test_github_json_request_rejects_invalid_method():
    with pytest.raises(RuntimeError, match="path is invalid"):
        github_json_request("TRACE", "/repos/x/y", "tok")


def test_github_json_request_rejects_relative_path():
    with pytest.raises(RuntimeError, match="path is invalid"):
        github_json_request("GET", "repos/x/y", "tok")


def test_github_json_request_rejects_double_slash_path():
    with pytest.raises(RuntimeError, match="path is invalid"):
        github_json_request("GET", "//evil.example.com/x", "tok")


def test_github_json_request_rejects_control_char_in_path():
    with pytest.raises(RuntimeError, match="path is invalid"):
        github_json_request("GET", "/repos/x\r\nHost: evil", "tok")


def test_github_json_request_rejects_oversized_path():
    with pytest.raises(RuntimeError, match="path is invalid"):
        github_json_request("GET", "/" + ("a" * 5000), "tok")


def test_github_json_request_rejects_empty_token():
    with pytest.raises(RuntimeError, match="credential is invalid"):
        github_json_request("GET", "/repos/x/y", "")


def test_github_json_request_rejects_oversized_token():
    with pytest.raises(RuntimeError, match="credential is invalid"):
        github_json_request("GET", "/repos/x/y", "a" * 70_000)


def test_github_json_request_rejects_control_char_token():
    with pytest.raises(RuntimeError, match="credential is invalid"):
        github_json_request("GET", "/repos/x/y", "tok\nen")


def test_github_json_request_rejects_oversized_payload():
    huge = {"data": "x" * (MAX_REQUEST_BYTES + 10)}
    with pytest.raises(RuntimeError, match="exceeds its safe size boundary"):
        github_json_request("POST", "/repos/x/y", "tok", payload=huge)


def test_github_json_request_success_returns_status_and_json():
    connection, _ = _mock_connection(status=200, body=b'{"a": 1}')
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        status, value = github_json_request("GET", "/repos/x/y", "tok")
    assert status == 200
    assert value == {"a": 1}
    connection.close.assert_called_once()


def test_github_json_request_rejects_redirect_status():
    connection, _ = _mock_connection(status=302, body=b"")
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        with pytest.raises(RuntimeError, match="redirect was rejected"):
            github_json_request("GET", "/repos/x/y", "tok")


def test_github_json_request_rejects_declared_oversized_response():
    connection, _ = _mock_connection(
        status=200, body=b"{}", content_length=str(MAX_RESPONSE_BYTES + 1)
    )
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        with pytest.raises(RuntimeError, match="exceeds its safe size boundary"):
            github_json_request("GET", "/repos/x/y", "tok")


def test_github_json_request_rejects_invalid_declared_length():
    connection, _ = _mock_connection(status=200, body=b"{}", content_length="not-a-number")
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        with pytest.raises(RuntimeError, match="response length is invalid"):
            github_json_request("GET", "/repos/x/y", "tok")


def test_github_json_request_rejects_oversized_actual_body():
    oversized = b"a" * (MAX_RESPONSE_BYTES + 2)
    connection, response = _mock_connection(status=200, body=oversized, content_length="0")
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        with pytest.raises(RuntimeError, match="exceeds its safe size boundary"):
            github_json_request("GET", "/repos/x/y", "tok")


def test_github_json_request_returns_empty_dict_for_empty_body():
    connection, _ = _mock_connection(status=204, body=b"", content_length="0")
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        status, value = github_json_request("GET", "/repos/x/y", "tok")
    assert status == 204
    assert value == {}


def test_github_json_request_rejects_invalid_json_body():
    connection, _ = _mock_connection(status=200, body=b"not json")
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        with pytest.raises(RuntimeError, match="response was invalid"):
            github_json_request("GET", "/repos/x/y", "tok")


def test_github_json_request_returns_empty_dict_for_non_dict_json():
    connection, _ = _mock_connection(status=200, body=b"[1, 2, 3]")
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        status, value = github_json_request("GET", "/repos/x/y", "tok")
    assert status == 200
    assert value == {}


def test_github_json_request_wraps_connection_errors():
    connection = MagicMock()
    connection.request.side_effect = OSError("boom")
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        with pytest.raises(RuntimeError, match="service is unavailable"):
            github_json_request("GET", "/repos/x/y", "tok")
    connection.close.assert_called_once()


def test_github_json_request_encodes_payload_as_compact_json():
    connection, response = _mock_connection(status=201, body=b'{"created": true}')
    with patch("github_agent.github_http.http.client.HTTPSConnection", return_value=connection):
        status, value = github_json_request(
            "POST", "/repos/x/y/issues", "tok", payload={"title": "hi"}
        )
    assert status == 201
    assert value == {"created": True}
    sent_kwargs = connection.request.call_args.kwargs
    assert sent_kwargs["body"] == json.dumps({"title": "hi"}, separators=(",", ":")).encode()
