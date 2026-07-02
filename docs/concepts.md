# Concept Registry — github-agent

> **Prefix**: `CONCEPT:GH-*`
> **Version**: 0.14.0
> **Bridge**: [`CONCEPT:ECO-4.0`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:GH-001` | Action Operations | MCP tool domain `action` — Action-routed dynamic tool registration |
| `CONCEPT:GH-002` | Branch Operations | MCP tool domain `branch` — Action-routed dynamic tool registration |
| `CONCEPT:GH-003` | Collaborator Operations | MCP tool domain `collaborator` — Action-routed dynamic tool registration |
| `CONCEPT:GH-004` | Commit Operations | MCP tool domain `commit` — Action-routed dynamic tool registration |
| `CONCEPT:GH-005` | Content Operations | MCP tool domain `content` — Action-routed dynamic tool registration |
| `CONCEPT:GH-006` | Issue Operations | MCP tool domain `issue` — Action-routed dynamic tool registration |
| `CONCEPT:GH-007` | Org Operations | MCP tool domain `org` — Action-routed dynamic tool registration |
| `CONCEPT:GH-008` | Pull Operations | MCP tool domain `pull` — Action-routed dynamic tool registration |
| `CONCEPT:GH-009` | Release Operations | MCP tool domain `release` — Action-routed dynamic tool registration |
| `CONCEPT:GH-010` | Repo Operations | MCP tool domain `repo` — Action-routed dynamic tool registration |
| `CONCEPT:GH-011` | Search & Discovery | MCP tool domain `search` — Action-routed dynamic tool registration |
| `CONCEPT:GH-012` | GraphQL Operations | MCP tool domain `graphql` (`GRAPHQLTOOL`) — native GitHub GraphQL client + schema discovery; one aliased query fans out across many repos (e.g. fleet-wide CI status in a single call) |

## Cross-Project References (from agent-utilities)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:ECO-4.0` | Unified Toolkit Ingestion | agent-utilities |
| `CONCEPT:ORCH-1.2` | Confidence-Gated Router | agent-utilities |
| `CONCEPT:OS-5.1` | Prompt Injection Defense | agent-utilities |
| `CONCEPT:OS-5.2` | Cognitive Scheduler | agent-utilities |
| `CONCEPT:OS-5.3` | Guardrail Engine | agent-utilities |
| `CONCEPT:OS-5.4` | Audit Logging | agent-utilities |
| `CONCEPT:KG-2.0` | Knowledge Graph Core | agent-utilities |

## Synergy with agent-utilities

This project integrates with `agent-utilities` via `CONCEPT:ECO-4.0` (Unified Toolkit Ingestion). The `github_agent` MCP server registers its tools with the agent-utilities FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all GH-* concepts.
