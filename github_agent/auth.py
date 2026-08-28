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


def _require_delegation_user_token() -> str:
    """Fetch the caller's OIDC user token for delegation, or raise."""
    user_token = getattr(local, "user_token", None)
    if not user_token:
        logger.error("No user token available for delegation")
        raise ValueError("No user token available for delegation")
    return user_token


def _validate_delegation_config(config: dict) -> tuple[str, str, str, str, str]:
    """Validate the OAuth delegation settings on ``config`` and return them."""
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

    return token_endpoint, client_id, client_secret, audience, delegated_scopes


def _exchange_delegated_token(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    audience: str,
    delegated_scopes: str,
    user_token: str,
) -> str:
    """Perform the OIDC token-exchange call and return the new bearer token."""
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
        return new_token
    except Exception as e:
        logger.error("Token exchange failed: error_type=%s", type(e).__name__)
        raise RuntimeError("Token exchange failed") from e
    finally:
        token_tls.cleanup()


def _build_delegated_client(instance: str, new_token: str, profile: ResolvedTLSProfile) -> Api:
    """Build the Api client from an exchanged delegated token."""
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


def _build_fixed_client(instance: str, token: str | None, profile: ResolvedTLSProfile) -> Api:
    """Build the Api client from a fixed configured credential."""
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

    if not config.get("enable_delegation"):
        return _build_fixed_client(instance, token, profile)

    user_token = _require_delegation_user_token()
    token_endpoint, client_id, client_secret, audience, delegated_scopes = (
        _validate_delegation_config(config)
    )
    new_token = _exchange_delegated_token(
        token_endpoint, client_id, client_secret, audience, delegated_scopes, user_token
    )
    return _build_delegated_client(instance, new_token, profile)


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
