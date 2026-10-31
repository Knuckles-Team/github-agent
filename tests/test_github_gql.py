"""GitHub GraphQL client — endpoint derivation + construction (no network)."""

from __future__ import annotations

import pytest
from scion_sdk.exceptions import MissingParameterError

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
    """close() on a never-used client is a safe no-op."""
    GraphQL(url="https://api.github.com", token="tok123").close()


@pytest.mark.spec("CONN-GITHUB-AGENT-CONNECTOR-IDENTITY-R002")
def test_wrapper_delegates_to_shared_graphql_client():
    from unittest.mock import patch

    from scion_sdk.tls.resolve import resolve_tls_profile
    from scion_sdk.transports.graphql import GraphQLClient

    profile = resolve_tls_profile("github")
    g = GraphQL(
        url="https://github.example/api/v3", token="tok123", tls_profile=profile  # sanitizer:ignore - synthetic test fixture token
    )
    assert isinstance(g._gql, GraphQLClient)
    assert g._gql.endpoint == "https://github.example/api/graphql"
    assert g._gql.headers["Authorization"] == "Bearer tok123"
    assert g._gql.tls_profile is profile
    with patch.object(GraphQLClient, "execute", return_value={"viewer": {}}) as ex:
        assert g.execute_gql("query { viewer { login } }", {"a": 1}, "Op") == {
            "viewer": {}
        }
    ex.assert_called_once_with("query { viewer { login } }", {"a": 1}, "Op")


def test_execute_failure_message_is_redacted_class_name():
    from unittest.mock import patch

    from scion_sdk.exceptions import ParameterError
    from scion_sdk.transports.graphql import GraphQLClient

    g = GraphQL(url="https://api.github.com", token="tok123")
    with patch.object(GraphQLClient, "_attempt", side_effect=ValueError("boom")):
        with pytest.raises(ParameterError, match="Query execution failed: ValueError"):
            g.execute_gql("query { viewer { login } }")
