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


def github_json_request(
    method: str,
    path: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Issue one bounded request directly to the fixed GitHub API origin."""

    if (
        method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}
        or not path.startswith("/")
        or path.startswith("//")
        or len(path) > 4_096
        or any(character in path for character in "\\\r\n#")
    ):
        raise RuntimeError("GitHub request path is invalid")
    if (
        not token
        or len(token) > 65_536
        or any(ord(character) < 32 for character in token)
    ):
        raise RuntimeError("GitHub credential is invalid")
    body = (
        None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    )
    if body is not None and len(body) > MAX_REQUEST_BYTES:
        raise RuntimeError("GitHub request exceeds its safe size boundary")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "github-agent-skill/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    connection = http.client.HTTPSConnection(
        GITHUB_API_HOST,
        443,
        timeout=30,
        context=_tls_context(),
    )
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
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
        if not raw:
            return response.status, {}
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError):
            raise RuntimeError("GitHub response was invalid") from None
        return response.status, value if isinstance(value, dict) else {}
    except (OSError, ssl.SSLError, http.client.HTTPException):
        raise RuntimeError("GitHub service is unavailable") from None
    finally:
        connection.close()
