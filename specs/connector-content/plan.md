# CONN-GITHUB-AGENT-CONNECTOR-CONTENT — Implementation plan

Governing spec: [spec.md](spec.md).
Code: ../../github_agent/mcp_server.py ../../github_agent/api/api_client_pulls.py ../../github_agent/github_gql.py

## Summary

`pyproject.toml` declares `github-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `github_agent` from it. Rollup `R007` adds one tool per GitHub API operation that autonomous delivery needs, plus a webhook normalizer. Each tool is a public method of the `Api` class. The SDK registers one tool per method, so no condensed action tool is added.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp, the REST and GraphQL clients in `github_agent/`
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest with recorded GitHub replies; served tests through an in-process MCP client
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: one GitHub call per tool call; pages of 100 items at most
- **Constraints**: one tool per API operation; no merge without an owner approval reference
- **Scale/Scope**: 12 new operation tools and one normalizer

## Constitution Check

- One owner per concept: pass. The connector owns GitHub calls only; records come from the SDK and verdicts from the engine.
- Connectors depend on the SDK and are called through arbora: pass. Every tool is reached through `fleet.call`.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
github_agent/api/api_client_pulls.py           existing  review, thread, arm, close and ready methods
github_agent/api/api_client_commits.py         existing  check runs and combined status
github_agent/api/api_client_notifications.py   new       notification polling
github_agent/github_gql.py                      existing  GraphQL queries and mutations
github_agent/scm_records.py                    new       reply to record mappers
github_agent/scm_events.py                     new       webhook normalizer
tests/test_change_proposal_tools.py            new       per-tool tests
tests/test_change_proposal_served.py           new       served proof
```

## Phase 0: Research

No unknowns. Each operation maps to one documented GitHub endpoint or mutation.

## Phase 1: Design

### Change-proposal tools

`register_tool_surface` at `github_agent/mcp_server.py:3669` registers one tool per public `Api` method. A new method therefore becomes a new tool named `github_<method>`. The GraphQL operations live in `GraphQL` (`github_agent/github_gql.py:13`). The arm tool extends the existing `enable_pull_request_auto_merge` mutation with the expected head commit. Mappers in `github_agent/scm_records.py` turn each reply into the SDK records. The design data is in [data-model.md](data-model.md).

### Safety

The arm tool declares the owner-approval flag of the SDK. It refuses without an approval reference. The platform principal never calls it directly; arbora calls it under an owner approval receipt.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first. The mappers come before the tools, and the served proof comes last.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: ruff check and format, `connector-standard`, `secrets`, `spec` and `tests-fast`. Each tool has one recorded-reply test. The served proof runs in the fast test set.

## Production caller chain

1. Arbora `fleet.call` reaches the `github-mcp` server over MCP.
2. `register_tool_surface` (`github_agent/mcp_server.py:3669`) registered each operation tool.
3. The tool calls the `Api` method, which calls the REST client or `GraphQL.execute_gql`.
4. Arbora's SCM intake calls `GithubEventNormalizer.normalize` for each webhook (row `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.13`).

## Cross-repo seams

- The change-proposal records and the normalizer protocol come from the SDK. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.13`. Contract: [change-proposal-contract](contracts/change-proposal-contract.md).
- Arbora calls the tools and the normalizer for delivery. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.8`, `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.12`. Contract: [delivery-calls](contracts/delivery-calls.md).
- The engine owns the classes the records map to. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.14`. Contract: [delivery-classes](contracts/delivery-classes.md).
