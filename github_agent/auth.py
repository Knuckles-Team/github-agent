#!/usr/bin/python

import os
import threading

import requests
from agent_utilities.base_utilities import get_logger, to_boolean
from agent_utilities.exceptions import AuthError, UnauthorizedError

from github_agent.api_client import Api

local = threading.local()
logger = get_logger(__name__)


def get_client(
    instance: str = os.getenv("GITHUB_URL", "https://api.github.com"),
    token: str | None = os.getenv("GITHUB_TOKEN", None),
    verify: bool = to_boolean(string=os.getenv("GITHUB_VERIFY", "True")),
    config: dict | None = None,
) -> Api:
    """
    Factory function to create the GitHub Api client.
    Supports fixed credentials (token) and delegation (OAuth exchange).
    """
    if config is None:
        from agent_utilities.mcp_utilities import config as default_config

        config = default_config

    if config.get("enable_delegation"):
        user_token = getattr(local, "user_token", None)
        if not user_token:
            logger.error("No user token available for delegation")
            raise ValueError("No user token available for delegation")

        logger.info(
            "Initiating OAuth token exchange for GitHub",
            extra={
                "audience": config["audience"],
                "scopes": config["delegated_scopes"],
            },
        )

        exchange_data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": user_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "audience": config["audience"],
            "scope": config["delegated_scopes"],
        }
        auth = (config["oidc_client_id"], config["oidc_client_secret"])
        try:
            response = requests.post(
                config["token_endpoint"], data=exchange_data, auth=auth
            )
            response.raise_for_status()
            new_token = response.json()["access_token"]
            logger.info("Token exchange successful")
        except Exception as e:
            logger.error(f"Token exchange failed: {str(e)}")
            raise RuntimeError(f"Token exchange failed: {str(e)}")

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
