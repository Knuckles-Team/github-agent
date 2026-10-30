# Concept Registry — github-agent

> **Prefix**: `CONCEPT:GH-*`
> **Version**: 0.14.0
> **Bridge**: [`CONCEPT:AU-ECO.messaging.native-backend-abstraction`](https://github.com/Knuckles-Team/scion-sdk/blob/main/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:GH-OS.governance.gh` | Action Operations | MCP tool domain `action` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-2` | Branch Operations | MCP tool domain `branch` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-3` | Collaborator Operations | MCP tool domain `collaborator` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-4` | Commit Operations | MCP tool domain `commit` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-5` | Content Operations | MCP tool domain `content` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-6` | Issue Operations | MCP tool domain `issue` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-7` | Org Operations | MCP tool domain `org` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-8` | Pull Operations | MCP tool domain `pull` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-9` | Release Operations | MCP tool domain `release` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-10` | Repo Operations | MCP tool domain `repo` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-11` | Search & Discovery | MCP tool domain `search` — Action-routed dynamic tool registration |
| `CONCEPT:GH-OS.governance.gh-12` | GraphQL Operations | MCP tool domain `graphql` (`GRAPHQLTOOL`) — native GitHub GraphQL client + schema discovery; one aliased query fans out across multiple repos (e.g. fleet-wide CI status in a single call) |

## Cross-Project References (from scion-sdk)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:AU-ECO.messaging.native-backend-abstraction` | Unified Toolkit Ingestion | scion-sdk |
| `CONCEPT:AU-ORCH.adapter.hot-cache-invalidation` | Confidence-Gated Router | scion-sdk |
| `CONCEPT:AU-OS.config.secrets-authentication` | Prompt Injection Defense | scion-sdk |
| `CONCEPT:AU-OS.state.cognitive-scheduler-preemption` | Cognitive Scheduler | scion-sdk |
| `CONCEPT:AU-OS.governance.reactive-multi-axis-budget` | Guardrail Engine | scion-sdk |
| `CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox` | Audit Logging | scion-sdk |
| `CONCEPT:AU-KG.query.object-graph-mapper` | Knowledge Graph Core | scion-sdk |

## Synergy with scion-sdk

This project integrates with `scion-sdk` via `CONCEPT:AU-ECO.messaging.native-backend-abstraction` (Unified Toolkit Ingestion). The `github_agent` MCP server registers its tools with the scion-sdk FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all GH-* concepts.
