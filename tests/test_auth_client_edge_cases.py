import os
import time
from unittest.mock import ANY, MagicMock, patch

import pytest
import requests
from agent_connector_sdk.auth.delegation import DelegationSettings
from agent_connector_sdk.auth.tokens import AccessToken
from agent_connector_sdk.exceptions import (
    AuthError,
    MissingParameterError,
    UnauthorizedError,
)

from github_agent.api_client import Api
from github_agent.auth import get_client

_ACCESS_TOKEN = AccessToken(
    value="mock-exchanged-github-token",
    ttl_seconds=3600.0,
    expires_at=time.monotonic() + 3600.0,
)

_DELEGATION_SETTINGS = DelegationSettings(
    enabled=True,
    token_endpoint="https://auth.github.com/token",
    client_id="client-id",
    client_secret_ref="env://GITHUB_OIDC_CLIENT_SECRET",
    audience="github-audience",
    scopes="repo,user",
)


def test_api_client_init_edge_cases():
    # 1. url=None raising MissingParameterError
    with pytest.raises(MissingParameterError):
        Api(url=None)

    # 2. debug=True sets level to logging.DEBUG
    with patch("requests.Session.get") as mock_get:
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        api = Api(debug=True, token="test-token")
        assert api.debug is True

        # 3. Every client receives a mandatory-verification TLS profile.
        api_profiled = Api(token="test-token")
        assert api_profiled.tls_profile.verify_enabled is True

        # 4. Omitted token warning
        with patch("github_agent.api_client.logger.warning") as mock_warn:
            Api(token=None)
            mock_warn.assert_called_with("No token provided for GitHub API")


def test_api_client_auth_errors():
    # 1. 401 status code raises AuthError
    with patch("requests.Session.get") as mock_get:
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized access"
        mock_get.return_value = mock_resp

        with pytest.raises(AuthError):
            Api(token="bad-token")

    # 2. 403 status code raises UnauthorizedError
    with patch("requests.Session.get") as mock_get:
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden access"
        mock_get.return_value = mock_resp

        with pytest.raises(UnauthorizedError):
            Api(token="bad-token")

    # 3. requests.exceptions.RequestException caught and logged
    with patch(
        "requests.Session.get",
        side_effect=requests.exceptions.RequestException("connection timed out"),
    ):
        with patch("github_agent.api_client.logger.error") as mock_err:
            Api(token="any-token")
            mock_err.assert_called_with(
                "Operation failed: error_type=%s", "RequestException"
            )


def test_auth_get_client_fixed_credentials_failure():
    # Test fixed credentials failure (Api raising AuthError)
    with patch("github_agent.auth.Api", side_effect=AuthError("Invalid credentials")):
        with pytest.raises(
            RuntimeError,
            match="AUTHENTICATION ERROR: The GitHub credentials provided are not valid",
        ):
            get_client()


def test_auth_get_client_delegation_missing_token():
    # delegation enabled, but the SDK has no verified caller token to delegate
    with (
        patch.object(
            DelegationSettings, "from_settings", return_value=_DELEGATION_SETTINGS
        ),
        patch("github_agent.auth.current_user_token", return_value=None),
    ):
        with pytest.raises(RuntimeError, match="^Token exchange failed$"):
            get_client()


def test_auth_get_client_delegation_exchange_failure():
    # delegation enabled, user token present, but token exchange fails
    with (
        patch.object(
            DelegationSettings, "from_settings", return_value=_DELEGATION_SETTINGS
        ),
        patch("github_agent.auth.current_user_token", return_value="caller-token"),
        patch(
            "github_agent.auth.exchange_token",
            side_effect=requests.exceptions.RequestException("OAuth connection error"),
        ),
    ):
        with pytest.raises(RuntimeError, match="^Token exchange failed$"):
            get_client()


def test_auth_get_client_delegation_auth_error():
    # delegation enabled, user token present, token exchange succeeds, but Api throws AuthError
    with (
        patch.object(
            DelegationSettings, "from_settings", return_value=_DELEGATION_SETTINGS
        ),
        patch("github_agent.auth.current_user_token", return_value="caller-token"),
        patch("github_agent.auth.exchange_token", return_value=_ACCESS_TOKEN),
        patch(
            "github_agent.auth.Api", side_effect=AuthError("Invalid exchanged token")
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="AUTHENTICATION ERROR: The delegated GitHub credentials are not valid",
        ):
            get_client()


def test_auth_get_client_delegation_success():
    # delegation enabled, user token present, token exchange succeeds, Api succeeds
    with (
        patch.object(
            DelegationSettings, "from_settings", return_value=_DELEGATION_SETTINGS
        ),
        patch("github_agent.auth.current_user_token", return_value="caller-token"),
        patch("github_agent.auth.exchange_token", return_value=_ACCESS_TOKEN),
        patch("github_agent.auth.Api") as mock_api_class,
    ):
        mock_api_instance = MagicMock()
        mock_api_class.return_value = mock_api_instance

        client = get_client()
        assert client == mock_api_instance
        mock_api_class.assert_called_with(
            url="https://api.github.com",
            token="mock-exchanged-github-token",
            tls_profile=ANY,
        )


def test_auth_get_client_default_config():
    # Test the fixed-credentials path (no delegation configured)
    with patch("github_agent.auth.Api") as mock_api_class:
        mock_api_instance = MagicMock()
        mock_api_class.return_value = mock_api_instance
        with patch.dict(os.environ, {"GITHUB_TOKEN": "my-fixed-token"}):
            client = get_client()
            assert client == mock_api_instance
            mock_api_class.assert_called_with(
                url="https://api.github.com",
                token="my-fixed-token",
                tls_profile=ANY,
            )
