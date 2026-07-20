#!/usr/bin/python


import logging
from typing import Any

from agent_utilities.core.exceptions import MissingParameterError, ParameterError
from agent_utilities.core.transport_security import (
    ResolvedTLSProfile,
    resolve_configured_tls_profile,
)
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport


class GraphQL:
    """Interact with GitHub's GraphQL API (parity with the REST ``Api`` client).

    GraphQL lets a single request fan out across many repositories via aliased
    sub-queries — e.g. fetch the latest-commit CI rollup for the whole fleet in
    one call instead of one REST request per repo.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        tls_profile: ResolvedTLSProfile | None = None,
        debug: bool = False,
    ):
        if not url:
            raise MissingParameterError("URL is required")
        if not token:
            raise MissingParameterError("Token is required")

        self.url = self._graphql_endpoint(url)
        self.token = token
        self.tls_profile = tls_profile or resolve_configured_tls_profile("github")
        self.debug = debug

        logging.basicConfig(
            level=logging.DEBUG if debug else logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        # GitHub GraphQL authenticates with a bearer token (classic PAT,
        # fine-grained PAT, or OAuth/GitHub-App installation token).
        self.headers = {"Authorization": f"Bearer {token}"}
        self.transport = RequestsHTTPTransport(
            url=self.url,
            headers=self.headers,
            **self.tls_profile.requests_kwargs(),
        )
        self.client = Client(
            transport=self.transport, fetch_schema_from_transport=False
        )

    def close(self) -> None:
        """Release GraphQL transport state and runtime-only TLS material."""
        self.client.close_sync()
        self.tls_profile.cleanup()

    @staticmethod
    def _graphql_endpoint(url: str) -> str:
        """Derive the GraphQL endpoint from the REST base URL.

        github.com → ``https://api.github.com/graphql``; GitHub Enterprise
        (``https://host/api/v3``) → ``https://host/api/graphql``.
        """
        base = url.rstrip("/")
        if base.endswith("/api/v3"):
            return base[: -len("/api/v3")] + "/api/graphql"
        return f"{base}/graphql"

    def execute_gql(
        self,
        query_str: str,
        variables: dict[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL query or mutation and return the raw data dict.

        Raises:
            ParameterError: If the query fails or the response carries errors.
        """
        try:
            query = gql(query_str)
            result = self.client.execute(
                query, variable_values=variables, operation_name=operation_name
            )
            if "errors" in result:
                raise ParameterError(f"GraphQL errors: {result['errors']}")
            return result
        except Exception as e:
            logging.error("GraphQL execution failed: error_type=%s", type(e).__name__)
            raise ParameterError(f"Query execution failed: {type(e).__name__}") from e

    def enable_pull_request_auto_merge(
        self, pull_request_id: str, merge_method: str = "MERGE"
    ) -> dict[str, Any]:
        """Enable auto-merge on a pull request (no REST equivalent exists).

        ``pull_request_id`` is the PR's GraphQL node id; ``merge_method`` is one
        of MERGE, SQUASH, or REBASE. Auto-merge queues the PR to merge once its
        required status checks pass and approvals are satisfied.
        """
        mutation = """
        mutation($pullRequestId: ID!, $mergeMethod: PullRequestMergeMethod!) {
          enablePullRequestAutoMerge(
            input: {pullRequestId: $pullRequestId, mergeMethod: $mergeMethod}
          ) {
            pullRequest {
              number
              autoMergeRequest { enabledAt mergeMethod }
            }
          }
        }
        """
        return self.execute_gql(
            mutation,
            {"pullRequestId": pull_request_id, "mergeMethod": merge_method.upper()},
        )

    def disable_pull_request_auto_merge(self, pull_request_id: str) -> dict[str, Any]:
        """Disable a previously-enabled auto-merge on a pull request (GraphQL)."""
        mutation = """
        mutation($pullRequestId: ID!) {
          disablePullRequestAutoMerge(input: {pullRequestId: $pullRequestId}) {
            pullRequest { number }
          }
        }
        """
        return self.execute_gql(mutation, {"pullRequestId": pull_request_id})
