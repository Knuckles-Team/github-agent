import logging
import os
import pytest
import requests
import urllib3
from unittest.mock import MagicMock, patch
from agent_utilities.exceptions import (
    AuthError,
    UnauthorizedError,
    MissingParameterError,
)
from github_agent.api_client import Api
from github_agent.auth import get_client, local


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

        # 3. verify=False disables InsecureRequestWarning
        with patch("urllib3.disable_warnings") as mock_disable:
            api_no_verify = Api(verify=False, token="test-token")
            mock_disable.assert_called_once_with(
                urllib3.exceptions.InsecureRequestWarning
            )
            assert api_no_verify.verify is False

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
            api = Api(token="any-token")
            mock_err.assert_called_with("Connection Error: connection timed out")


def test_auth_get_client_fixed_credentials_failure():
    # Test fixed credentials failure (Api raising AuthError)
    with patch("github_agent.auth.Api", side_effect=AuthError("Invalid credentials")):
        with pytest.raises(
            RuntimeError,
            match="AUTHENTICATION ERROR: The GitHub credentials provided are not valid",
        ):
            get_client({"enable_delegation": False})


def test_auth_get_client_delegation_missing_token():
    # delegation enabled, but local.user_token is missing
    if hasattr(local, "user_token"):
        delattr(local, "user_token")

    with pytest.raises(ValueError, match="No user token available for delegation"):
        get_client({"enable_delegation": True})


def test_auth_get_client_delegation_exchange_failure():
    # delegation enabled, user token present, but token exchange fails
    local.user_token = "some-subject-token"
    config = {
        "enable_delegation": True,
        "audience": "github-audience",
        "delegated_scopes": "repo,user",
        "token_endpoint": "https://auth.github.com/token",
        "oidc_client_id": "client-id",
        "oidc_client_secret": "client-secret",
    }

    with patch(
        "requests.post",
        side_effect=requests.exceptions.RequestException("OAuth connection error"),
    ):
        with pytest.raises(
            RuntimeError, match="Token exchange failed: OAuth connection error"
        ):
            get_client(config)


def test_auth_get_client_delegation_auth_error():
    # delegation enabled, user token present, token exchange succeeds, but Api throws AuthError
    local.user_token = "some-subject-token"
    config = {
        "enable_delegation": True,
        "audience": "github-audience",
        "delegated_scopes": "repo,user",
        "token_endpoint": "https://auth.github.com/token",
        "oidc_client_id": "client-id",
        "oidc_client_secret": "client-secret",
    }

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"access_token": "exchanged-github-token"}

    with patch("requests.post", return_value=mock_resp):
        with patch(
            "github_agent.auth.Api", side_effect=AuthError("Invalid exchanged token")
        ):
            with pytest.raises(
                RuntimeError,
                match="AUTHENTICATION ERROR: The delegated GitHub credentials are not valid",
            ):
                get_client(config)


def test_auth_get_client_delegation_success():
    # delegation enabled, user token present, token exchange succeeds, Api succeeds
    local.user_token = "some-subject-token"
    config = {
        "enable_delegation": True,
        "audience": "github-audience",
        "delegated_scopes": "repo,user",
        "token_endpoint": "https://auth.github.com/token",
        "oidc_client_id": "client-id",
        "oidc_client_secret": "client-secret",
    }

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"access_token": "exchanged-github-token"}

    with patch("requests.post", return_value=mock_resp):
        with patch("github_agent.auth.Api") as mock_api_class:
            mock_api_instance = MagicMock()
            mock_api_class.return_value = mock_api_instance

            client = get_client(config)
            assert client == mock_api_instance
            mock_api_class.assert_called_with(
                url="https://api.github.com",
                token="exchanged-github-token",
                verify=True,
            )


def test_auth_get_client_default_config():
    # Test that config=None defaults properly
    with patch("github_agent.auth.Api") as mock_api_class:
        mock_api_instance = MagicMock()
        mock_api_class.return_value = mock_api_instance
        with patch.dict(os.environ, {"GITHUB_TOKEN": "my-fixed-token"}):
            client = get_client(None)
            assert client == mock_api_instance
            mock_api_class.assert_called_with(
                url="https://api.github.com", token="my-fixed-token", verify=True
            )


def test_auth_invalid_oauth_types():
    # delegation enabled, user token present, but one of the parameters is not a string
    local.user_token = "some-subject-token"
    config = {
        "enable_delegation": True,
        "audience": 12345,  # Invalid type (must be string)
        "delegated_scopes": "repo,user",
        "token_endpoint": "https://auth.github.com/token",
        "oidc_client_id": "client-id",
        "oidc_client_secret": "client-secret",
    }
    with pytest.raises(ValueError, match="Invalid OAuth configuration parameters"):
        get_client(config)
