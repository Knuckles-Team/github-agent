# CONN-GITHUB-AGENT-CONNECTOR-CONTENT — The connector serves its packaged content

Owner: github-agent. Cross-repo IDs: `SCION-CONNECTOR-CONTROL-R071`, `SCION-CONNECTOR-CONTROL-R073`, `SCION-GOVERNED-WRITEBACK-R004.8`, `MEDULLA-CQ-CONN-R086`.
Component: Connector

The `github-mcp` server serves its packaged skills, prompts and ontology to any client, with no wiring of its own. It also serves one tool per GitHub API operation that autonomous delivery of a change proposal needs.

## User Scenarios & Testing

### User Story 1 - Packaged content and ingestible sources (Priority: P1)

A client connects to the `github-mcp` server and reads its skills, prompts, ontology and ingest sources with no extra wiring.

**Acceptance Scenarios**:

1. **Given** a fresh server, **When** a client lists resources and prompts, **Then** the packaged `skill://` and `ontology://` resources and each prompt appear.
2. **Given** the packaged ontology, **When** the certification runs, **Then** every class reaches a parent class and no shared class is redeclared.

### User Story 2 - Change-proposal delivery tools (Priority: P1)

The platform delivers a spec row as a pull request on its own. It reads reviews, threads and checks, answers review feedback and asks to arm auto-merge. Each step uses one tool per GitHub API operation.

**Acceptance Scenarios**:

1. **Given** an open pull request, **When** arbora calls the review, thread and check tools, **Then** each reply maps to one typed change-proposal record.
2. **Given** an arm request without an owner approval reference, **When** the tool runs, **Then** it refuses and arms nothing.
3. **Given** a signed webhook, **When** arbora passes it to the normalizer, **Then** one feedback event of one closed kind results.

### Edge Cases

- An arm request whose expected head differs from the current head refuses with `head_mismatch`.
- A resolve call on a resolved thread writes nothing new.
- A webhook of an unsupported event refuses with `SCM_EVENT_UNSUPPORTED`.
- A review reply over 65,536 characters refuses before any call.

## Requirements

- **FR-001**: The server serves its packaged content by default. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R001`.
- **FR-002**: The ontology joins the parent graph and keeps competency paths. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R002`, `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R004`.
- **FR-003**: The connector declares its ingest sources and access contracts. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003`, `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R005`.
- **FR-004**: The connector applies one conditional repository file effect. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R006`.
- **FR-005**: The connector serves one tool per change-proposal operation and a webhook normalizer. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007`.
- **FR-006**: An agent run writes only what its task asked for, proven by diffing repository state. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R008`.

## Success Criteria

- **SC-001**: Each of the twelve change-proposal operations has exactly one served tool. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.15`.
- **SC-002**: The arm tool refuses in 100% of calls without an approval reference. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.8`.
- **SC-003**: Each supported webhook maps to exactly one feedback kind. Rows: `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.13`.

## Assumptions

- The SDK provides the change-proposal records, the normalizer protocol and the approval flag of `SCION-SCM`.
- Arbora owns the webhook route, the admission of feedback and the approval receipt.
- The engine owns the delivery records and every merge verdict.

## Why a new spec

This spec is not new. The search `git grep -n -iE "pull request|review|check run|auto-merge" -- specs` found only this spec. It owns the served tools and content of the connector. The change-proposal tools extend it as rollup `R007`.

## Scope and interfaces

In scope: packaged content, ingest sources, access contracts, the repository file effect and the change-proposal tools.
Non-goals: the webhook route and admission (arbora), merge verdicts and delivery records (the engine), and the shared records (the SDK).

## Traceability

| Label | Rows | Design section | Test |
|---|---|---|---|
| FR-005 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007` | plan.md "Change-proposal tools" | `tests/test_change_proposal_served.py::test_served_tools_and_normalizer_round_trip` |
| FR-006 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R008` | plan.md "Side-effect audit" | `tests/test_side_effect_audit.py::test_injected_text_causes_no_write_beyond_the_intended_issue` |
