"""GitHub GraphQL client — endpoint derivation + construction (no network)."""

from __future__ import annotations

import pytest
from agent_utilities.core.exceptions import MissingParameterError

from github_agent.github_gql import GraphQL


def test_endpoint_github_com():
    assert GraphQL._graphql_endpoint("https://api.github.com") == (
        "https://api.github.com/graphql"
    )
    assert GraphQL._graphql_endpoint("https://api.github.com/") == (
        "https://api.github.com/graphql"
    )


def test_endpoint_enterprise():
    assert GraphQL._graphql_endpoint("https://github.example/api/v3") == (
        "https://github.example/api/graphql"
    )


def test_requires_url_and_token():
    with pytest.raises(MissingParameterError):
        GraphQL(url=None, token="x")
    with pytest.raises(MissingParameterError):
        GraphQL(url="https://api.github.com", token=None)


def test_construction_sets_bearer_header():
    g = GraphQL(url="https://api.github.com", token="tok123")
    assert g.headers["Authorization"] == "Bearer tok123"
    assert g.url.endswith("/graphql")


def test_close_without_session_does_not_raise():
    """Regression guard: under gql 4.0, Client.close_sync() unconditionally
    asserts ``self.session`` — an attribute only ever set by connect_sync()
    (called implicitly the first time a query executes). A GraphQL client
    that was constructed but never used to run a query has no such session
    yet, so close() must be a safe no-op rather than raising AttributeError.
    """
    g = GraphQL(url="https://api.github.com", token="tok123")
    g.close()


def test_close_after_session_opened_closes_cleanly():
    """Once a sync session has actually been opened, close() still closes it."""
    g = GraphQL(url="https://api.github.com", token="tok123")
    g.client.connect_sync()
    assert hasattr(g.client, "session")
    g.close()
