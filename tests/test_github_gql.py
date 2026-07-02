"""GitHub GraphQL client — endpoint derivation + construction (no network)."""

from __future__ import annotations

import pytest

from github_agent.github_gql import GraphQL


def test_endpoint_github_com():
    assert GraphQL._graphql_endpoint("https://api.github.com") == (
        "https://api.github.com/graphql"
    )
    assert GraphQL._graphql_endpoint("https://api.github.com/") == (
        "https://api.github.com/graphql"
    )


def test_endpoint_enterprise():
    assert GraphQL._graphql_endpoint("https://ghe.corp/api/v3") == (
        "https://ghe.corp/api/graphql"
    )


def test_requires_url_and_token():
    with pytest.raises(Exception):
        GraphQL(url=None, token="x")
    with pytest.raises(Exception):
        GraphQL(url="https://api.github.com", token=None)


def test_construction_sets_bearer_header():
    g = GraphQL(url="https://api.github.com", token="tok123")
    assert g.headers["Authorization"] == "Bearer tok123"
    assert g.url.endswith("/graphql")
