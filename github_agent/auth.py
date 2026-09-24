#!/usr/bin/python

import httpx
from agent_connector_sdk.auth.delegation import (
    DelegationSettings,
    current_user_token,
    exchange_token,
)
from agent_connector_sdk.config import setting
from agent_connector_sdk.exceptions import (
    AuthError,
    LoginRequiredError,
    UnauthorizedError,
)
from agent_connector_sdk.tls.profile import ResolvedTLSProfile
from agent_connector_sdk.tls.resolve import resolve_tls_profile
from agent_connector_sdk.utilities import get_logger

from github_agent.api_client import Api

logger = get_logger(__name__)


def allow_destructive_default() -> bool:
    """Fleet-wide default for the destructive-action gate.

    Destructive MCP actions (e.g. organization delete, member removal) are
    blocked unless the caller passes allow_destructive=true or the
    GITHUB_ALLOW_DESTRUCTIVE environment variable is set truthy.
    """
    return setting("GITHUB_ALLOW_DESTRUCTIVE", False)


def _exchange_delegated_token(settings: DelegationSettings) -> str:
    """Exchange the caller's verified MCP token for a downstream GitHub token."""
    subject_token = current_user_token()
    if not subject_token:
        raise LoginRequiredError("no verified caller token to delegate")
    logger.info(
        "Initiating OAuth token exchange for GitHub",
        extra={"audience": settings.audience, "scopes": settings.scopes},
    )
    try:
        with httpx.Client(timeout=30) as http_client:
            access_token = exchange_token(
                settings, subject_token=subject_token, http_client=http_client
            )
        logger.info("Token exchange successful")
        return access_token.value
    except Exception as e:
        logger.error("Token exchange failed: error_type=%s", type(e).__name__)
        raise RuntimeError("Token exchange failed") from e


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
    tls_profile: ResolvedTLSProfile | None = None,
) -> Api:
    """
    Factory function to create the GitHub Api client.
    Supports fixed credentials (token) and delegation (OAuth exchange).
    """
    instance = setting("GITHUB_URL", "https://api.github.com")
    token = setting("GITHUB_TOKEN", None)
    profile = tls_profile or resolve_tls_profile("github")

    settings = DelegationSettings.from_settings()
    if not settings.enabled:
        return _build_fixed_client(instance, token, profile)

    new_token = _exchange_delegated_token(settings)
    return _build_delegated_client(instance, new_token, profile)


def get_graphql_client(
    tls_profile: ResolvedTLSProfile | None = None,
):
    """Factory for the GitHub GraphQL client (parity with :func:`get_client`).

    Resolves the same ``GITHUB_URL`` / ``GITHUB_TOKEN`` / TLS profile and
    honours OIDC delegation, then returns a :class:`~github_agent.github_gql.GraphQL`.
    """
    from github_agent.github_gql import GraphQL

    instance = setting("GITHUB_URL", "https://api.github.com")
    token = setting("GITHUB_TOKEN", None)
    profile = tls_profile or resolve_tls_profile("github")

    settings = DelegationSettings.from_settings()
    if settings.enabled:
        # Reuse the REST factory's OIDC token exchange, then read back the
        # exchanged bearer token for the GraphQL transport.
        api = get_client(tls_profile=profile)
        authorization = str(api.headers.get("Authorization", ""))
        token = authorization.removeprefix("Bearer ") or token

    return GraphQL(url=instance, token=token, tls_profile=profile)
