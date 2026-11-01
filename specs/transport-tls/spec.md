# CONN-GITHUB-AGENT-TRANSPORT-TLS — Transport trust comes from the SDK TLS profile

Owner: github-agent. Cross-repo IDs: `SCION-CONNECTOR-CONTROL-R017`, `SCION-CENTRALIZED-CAPABILITIES-R008`.

## Purpose and user stories

An operator points this connector at a private CA, or at GitHub Enterprise behind one, through the shared TLS profile settings. Every HTTPS call the connector makes, the GraphQL client and the fixed-origin skill-script client alike, verifies the peer with that one profile.

## Why a new spec

`specs/connector-identity` owns only the fleet name. No other local spec owns transport or trust.

## Scope and interfaces

In scope: the rows of [requirements.md](requirements.md). Out of scope: the SDK modules and the engine, which own the shared capabilities.

## User Scenarios & Testing

### User Story 1 - Transport trust comes from the SDK TLS profile (Priority: P1)

An operator points this connector at a private CA, or at GitHub Enterprise behind one, through the shared TLS profile settings. Every HTTPS call the connector makes, the GraphQL client and the fixed-origin skill-script client alike, verifies the peer with that one profile.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001` (sdk tls profile) is exercised, **Then** the connection receives the profile's context, the profile is cleaned up, and no `ssl.create_default_context` call remains in `github_agent/`.

### Edge Cases

- If `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001` is not met, its proving test fails: sdk tls profile.

## Requirements

- **FR-001**: SDK TLS profile. Rows: `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001`.

## Success Criteria

- **SC-001**: The connection receives the profile's context, the profile is cleaned up, and no `ssl.create_default_context` call remains in `github_agent/`. Rows: `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |
