#!/usr/bin/python

import threading

import requests
from agent_utilities.base_utilities import get_logger
from agent_utilities.core.config import setting
from agent_utilities.exceptions import AuthError, UnauthorizedError

local = threading.local()
from github_agent.api_client import Api

logger = get_logger(__name__)


def allow_destructive_default() -> bool:
    """Fleet-wide default for the destructive-action gate.

    Destructive MCP actions (e.g. organization delete, member removal) are
    blocked unless the caller passes allow_destructive=true or the
    GITHUB_ALLOW_DESTRUCTIVE environment variable is set truthy.
    """
    return setting("GITHUB_ALLOW_DESTRUCTIVE", False)


def get_client(config: dict | None = None) -> Api:
    """
    Factory function to create the GitHub Api client.
    Supports fixed credentials (token) and delegation (OAuth exchange).
    """
    instance = setting("GITHUB_URL", "https://api.github.com")
    token = setting("GITHUB_TOKEN", None)
    if setting("GITHUB_SSL_VERIFY", None) is not None:
        verify = setting("GITHUB_SSL_VERIFY", True)
    else:
        verify = setting("GITHUB_VERIFY", True)

    if config is None:
        from agent_utilities.mcp_utilities import config as default_config

        config = default_config

    if config.get("enable_delegation"):
        user_token = getattr(local, "user_token", None)
        if not user_token:
            logger.error("No user token available for delegation")
            raise ValueError("No user token available for delegation")

        token_endpoint = config.get("token_endpoint")
        client_id = config.get("oidc_client_id")
        client_secret = config.get("oidc_client_secret")
        audience = config.get("audience")
        delegated_scopes = config.get("delegated_scopes")

        if (
            not isinstance(token_endpoint, str)
            or not isinstance(client_id, str)
            or not isinstance(client_secret, str)
            or not isinstance(audience, str)
            or not isinstance(delegated_scopes, str)
        ):
            raise ValueError("Invalid OAuth configuration parameters")

        logger.info(
            "Initiating OAuth token exchange for GitHub",
            extra={
                "audience": audience,
                "scopes": delegated_scopes,
            },
        )

        exchange_data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": user_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",  # nosec B105
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",  # nosec B105
            "audience": audience,
            "scope": delegated_scopes,
        }
        auth = (client_id, client_secret)
        try:
            response = requests.post(
                token_endpoint, data=exchange_data, auth=auth, timeout=30
            )
            response.raise_for_status()
            new_token = response.json()["access_token"]
            logger.info("Token exchange successful")
        except Exception as e:
            logger.error(f"Token exchange failed: {str(e)}")
            raise RuntimeError(f"Token exchange failed: {str(e)}") from e

        try:
            return Api(
                url=instance,
                token=new_token,
                verify=verify,
            )
        except (AuthError, UnauthorizedError) as e:
            raise RuntimeError(
                f"AUTHENTICATION ERROR: The delegated GitHub credentials are not valid for '{instance}'."
                f"Error details: {str(e)}"
            ) from e
    else:
        logger.info("Using fixed credentials for GitHub API")
        try:
            return Api(
                url=instance,
                token=token,
                verify=verify,
            )
        except (AuthError, UnauthorizedError) as e:
            raise RuntimeError(
                f"AUTHENTICATION ERROR: The GitHub credentials provided are not valid for '{instance}'. "
                f"Please check your GITHUB_TOKEN and GITHUB_URL environment variables. "
                f"Error details: {str(e)}"
            ) from e


def get_graphql_client(config: dict | None = None):
    """Factory for the GitHub GraphQL client (parity with :func:`get_client`).

    Resolves the same ``GITHUB_URL`` / ``GITHUB_TOKEN`` / verify settings and
    honours OIDC delegation, then returns a :class:`~github_agent.github_gql.GraphQL`.
    """
    from github_agent.github_gql import GraphQL

    instance = setting("GITHUB_URL", "https://api.github.com")
    token = setting("GITHUB_TOKEN", None)
    if setting("GITHUB_SSL_VERIFY", None) is not None:
        verify = setting("GITHUB_SSL_VERIFY", True)
    else:
        verify = setting("GITHUB_VERIFY", True)

    if config is None:
        from agent_utilities.mcp_utilities import config as default_config

        config = default_config

    if config.get("enable_delegation"):
        # Reuse the REST factory's OIDC token exchange, then read back the
        # exchanged bearer token for the GraphQL transport.
        api = get_client(config)
        token = getattr(api, "token", None) or token

    return GraphQL(url=instance, token=token, verify=verify)
