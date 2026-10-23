# CONN-GITHUB-AGENT-CONNECTOR-IDENTITY — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-GITHUB-AGENT-CONNECTOR-IDENTITY-R001` | `tests/test_fleet_name.py::test_connector_name_is_the_fleet_name` | Every connector name in the package is `github-mcp`. |
| T-002 | `CONN-GITHUB-AGENT-CONNECTOR-IDENTITY-R002` | `tests/test_github_gql.py::test_wrapper_delegates_to_shared_graphql_client` | GraphQL calls use the shared client with the GitHub endpoint, bearer header and TLS profile. |
