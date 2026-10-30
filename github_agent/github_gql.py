#!/usr/bin/python


import logging
from typing import Any

from scion_sdk.exceptions import MissingParameterError
from scion_sdk.tls.profile import ResolvedTLSProfile
from scion_sdk.tls.resolve import resolve_tls_profile
from scion_sdk.transports.graphql import GraphQLAuth, GraphQLClient


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
        self.tls_profile = tls_profile or resolve_tls_profile("github")
        self.debug = debug

        logging.basicConfig(
            level=logging.DEBUG if debug else logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        # GitHub GraphQL authenticates with a bearer token (classic PAT,
        # fine-grained PAT, or OAuth/GitHub-App installation token).
        self.headers = {"Authorization": f"Bearer {token}"}
        self._gql = GraphQLClient(
            self.url,
            auth=GraphQLAuth.bearer(token),
            tls_profile=self.tls_profile,
            tls_service="github",
        )

    @property
    def client(self) -> Any:
        """The underlying protocol client, built on first use."""
        return self._gql.client

    def close(self) -> None:
        """Release GraphQL transport state and runtime-only TLS material."""
        self._gql.close()

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
        return self._gql.execute(query_str, variables, operation_name)

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
