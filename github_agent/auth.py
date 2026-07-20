#!/usr/bin/python

import threading

import requests
from agent_utilities.base_utilities import get_logger
from agent_utilities.core.config import config as agent_config
from agent_utilities.core.config import setting
from agent_utilities.core.exceptions import AuthError, UnauthorizedError
from agent_utilities.core.transport_security import (
    ResolvedTLSProfile,
    resolve_configured_tls_profile,
)

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


def get_client(
    config: dict | None = None,
    tls_profile: ResolvedTLSProfile | None = None,
) -> Api:
    """
    Factory function to create the GitHub Api client.
    Supports fixed credentials (token) and delegation (OAuth exchange).
    """
    instance = setting("GITHUB_URL", "https://api.github.com")
    token = setting("GITHUB_TOKEN", None)
    profile = tls_profile or resolve_configured_tls_profile("github")

    if config is None:
        from agent_utilities.mcp.server_factory import mcp_auth_config as default_config

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
        token_tls = resolve_configured_tls_profile(
            "oauth2_token",
            profile_name=agent_config.oauth2_token_tls_profile,
            profile_ref=agent_config.oauth2_token_tls_profile_ref,
            config=agent_config,
        )
        try:
            response = requests.post(
                token_endpoint,
                data=exchange_data,
                auth=auth,
                timeout=30,
                **token_tls.requests_kwargs(),
            )
            response.raise_for_status()
            new_token = response.json()["access_token"]
            logger.info("Token exchange successful")
        except Exception as e:
            logger.error("Token exchange failed: error_type=%s", type(e).__name__)
            raise RuntimeError("Token exchange failed") from e
        finally:
            token_tls.cleanup()

        try:
            return Api(
                url=instance,
                token=new_token,
                tls_profile=profile,
            )
        except (AuthError, UnauthorizedError) as e:
            raise RuntimeError(
                "AUTHENTICATION ERROR: The delegated GitHub credentials are not valid."
            ) from e
    else:
        logger.info("Using fixed credentials for GitHub API")
        try:
            return Api(
                url=instance,
                token=token,
                tls_profile=profile,
            )
        except (AuthError, UnauthorizedError) as e:
            raise RuntimeError(
                "AUTHENTICATION ERROR: The GitHub credentials provided are not valid. "
                "Please check the configured credential and endpoint references."
            ) from e


def get_graphql_client(
    config: dict | None = None,
    tls_profile: ResolvedTLSProfile | None = None,
):
    """Factory for the GitHub GraphQL client (parity with :func:`get_client`).

    Resolves the same ``GITHUB_URL`` / ``GITHUB_TOKEN`` / TLS profile and
    honours OIDC delegation, then returns a :class:`~github_agent.github_gql.GraphQL`.
    """
    from github_agent.github_gql import GraphQL

    instance = setting("GITHUB_URL", "https://api.github.com")
    token = setting("GITHUB_TOKEN", None)
    profile = tls_profile or resolve_configured_tls_profile("github")

    if config is None:
        from agent_utilities.mcp.server_factory import mcp_auth_config as default_config

        config = default_config

    if config.get("enable_delegation"):
        # Reuse the REST factory's OIDC token exchange, then read back the
        # exchanged bearer token for the GraphQL transport.
        api = get_client(config, tls_profile=profile)
        authorization = str(api.headers.get("Authorization", ""))
        token = authorization.removeprefix("Bearer ") or token

    return GraphQL(url=instance, token=token, tls_profile=profile)
