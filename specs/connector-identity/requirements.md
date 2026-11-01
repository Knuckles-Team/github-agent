# CONN-GITHUB-AGENT-CONNECTOR-IDENTITY — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-GITHUB-AGENT-CONNECTOR-IDENTITY-R001` | **Fleet name.** `pyproject.toml` declares `github-mcp` as the single `scion_sdk.connector` entry point. The content pack, the manifest `connector`, the fingerprints and the ingest binding all carry `github-mcp`. Proving test: `tests/test_fleet_name.py::test_connector_name_is_the_fleet_name`. Caller chain: console_script github-mcp -> github_agent/mcp_server.py::mcp_server (L3692) -> get_mcp_instance (L3658) -> create_mcp_server with ConnectorContent.for_package('github_agent', ...); the scion_sdk.connector entry point is github_agent::connector (__init__.py L10). Shape: the fleet name github-mcp comes from pyproject.toml and scion_sdk.connector_identity; tests/test_fleet_name.py walks IngestBinding and ConnectorContent calls in github_agent for a connector keyword. | Every connector name in the package is `github-mcp`. |
| `CONN-GITHUB-AGENT-CONNECTOR-IDENTITY-R002` | **Shared GraphQL client.** The connector's GraphQL calls go through the SDK shared GraphQL client. It declares GraphQL only through the SDK `graphql` extra. Proving test: `tests/test_github_gql.py::test_wrapper_delegates_to_shared_graphql_client`. Caller chain: github_agent/auth.py::get_client (L117) -> github_agent/github_gql.py::GraphQL (L13) -> execute_gql (L74) -> scion_sdk.transports.graphql.GraphQLClient.execute. Shape: GraphQL wraps scion_sdk.transports.graphql.GraphQLClient with GraphQLAuth and resolve_tls_profile; pyproject.toml declares the SDK extra as scion-sdk[mcp,graphql]. | GraphQL calls use the shared client with the GitHub endpoint, bearer header and TLS profile. |
