# CONN-GITHUB-AGENT-TRANSPORT-TLS — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

`github_agent/github_http.py` and `github_agent/github_gql.py` both call `resolve_tls_profile("github")`. The profile carries the per-instance CA (`GITHUB_CA_BUNDLE`, `GITHUB_CA_BUNDLE_REF`, `GITHUB_CA_DIRECTORY`, or a named `GITHUB_TLS_PROFILE`), system trust, and the minimum TLS version. The skill-script client passes `profile.ssl_context` to its connection and calls `profile.cleanup()` when the request ends.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest; the proof level each row names
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: N/A
- **Constraints**: one tool per API operation; shared capabilities come from the SDK
- **Scale/Scope**: 1 rows

## Constitution Check

- One owner per concept: pass. The connector owns one external system's calls, mappings and tools only.
- Connectors depend on the SDK and are called through arbora: pass.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
github_agent/github_http.py  existing
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

The rows are delivered in place; the design is the caller chain below.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001`: skill scripts github_agent/skills/github-triage-resolver/scripts/gh_write.py and github-project-provisioner/scripts/enable_pages.py -> github_agent/github_http.py::github_json_request (L97) -> _tls_profile (L19) -> resolve_tls_profile('github').

## Cross-repo seams

none. The cited owner rows are library capabilities the connector calls in process; spec.md names them.
