"""Small fixed-origin HTTPS client for standalone GitHub skill scripts."""

from __future__ import annotations

import http.client
import json
import os
import ssl
from typing import Any

GITHUB_API_HOST = "api.github.com"
MAX_REQUEST_BYTES = 1 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _tls_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    capath = os.environ.get("SSL_CERT_DIR")
    try:
        return ssl.create_default_context(cafile=cafile or None, capath=capath or None)
    except (OSError, ssl.SSLError):
        raise RuntimeError("runtime TLS trust configuration is invalid") from None


def _validate_request_shape(method: str, path: str) -> None:
    """Reject a method/path pair that does not match the fixed-origin contract."""
    if (
        method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}
        or not path.startswith("/")
        or path.startswith("//")
        or len(path) > 4_096
        or any(character in path for character in "\\\r\n#")
    ):
        raise RuntimeError("GitHub request path is invalid")


def _validate_token_shape(token: str) -> None:
    """Reject a credential that cannot possibly be a valid GitHub token."""
    if (
        not token
        or len(token) > 65_536
        or any(ord(character) < 32 for character in token)
    ):
        raise RuntimeError("GitHub credential is invalid")


def _encode_request_body(payload: dict[str, Any] | None) -> bytes | None:
    """Encode the JSON payload and enforce the request size boundary."""
    body = (
        None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    )
    if body is not None and len(body) > MAX_REQUEST_BYTES:
        raise RuntimeError("GitHub request exceeds its safe size boundary")
    return body


def _request_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "github-agent-skill/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _read_bounded_response(response: http.client.HTTPResponse) -> bytes:
    """Read the response body, enforcing the declared and actual size caps."""
    declared = response.getheader("Content-Length")
    if declared:
        try:
            if int(declared) > MAX_RESPONSE_BYTES:
                raise RuntimeError("GitHub response exceeds its safe size boundary")
        except ValueError:
            raise RuntimeError("GitHub response length is invalid") from None
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("GitHub response exceeds its safe size boundary")
    if 300 <= response.status < 400:
        raise RuntimeError("GitHub redirect was rejected")
    return raw


def _decode_response_json(raw: bytes) -> dict[str, Any]:
    """Parse the response body, tolerating only a JSON object or emptiness."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        raise RuntimeError("GitHub response was invalid") from None
    return value if isinstance(value, dict) else {}


def github_json_request(
    method: str,
    path: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Issue one bounded request directly to the fixed GitHub API origin."""
    _validate_request_shape(method, path)
    _validate_token_shape(token)
    body = _encode_request_body(payload)
    headers = _request_headers(token)
    connection = http.client.HTTPSConnection(
        GITHUB_API_HOST,
        443,
        timeout=30,
        context=_tls_context(),
    )
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = _read_bounded_response(response)
        return response.status, _decode_response_json(raw)
    except (OSError, ssl.SSLError, http.client.HTTPException):
        raise RuntimeError("GitHub service is unavailable") from None
    finally:
        connection.close()
