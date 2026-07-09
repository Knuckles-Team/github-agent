---
name: github-dependabot-review
skill_type: skill
description: >-
  Review GitHub Dependabot vulnerability alerts for a single repository or an entire
  organization via the github-agent MCP server — list alerts (filter by severity, state,
  ecosystem, package, scope), drill into a single alert, and dismiss or reopen an alert with
  explicit confirmation. Use when the agent must audit dependency vulnerabilities, triage
  security alerts across an org/repo, or dismiss a known-acceptable alert. Do NOT use for code
  scanning / secret scanning alerts (not covered here), to gate a PR's merge on CI
  (github-actions-ci-review), or to open remediation issues (github-issue-tracking).
license: MIT
tags: [github, dependabot, security, vulnerabilities, alerts, dependencies, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# GitHub Dependabot Review

Domain-typed review of **GitHub Dependabot** vulnerability alerts via the github-agent MCP
server, scoped to a single **repository** or an entire **organization**. Surfaces open (and
dismissed/fixed) dependency alerts, drills into a single alert, and dismisses or reopens an
alert. Prefer this condensed tool over raw REST.

## When to use
- Audit a **repo's** Dependabot alerts, optionally filtered to `severity=critical` or
  `state=open`.
- Sweep every alert across an **org** in one endpoint (`list_org`).
- Read the full detail of one alert (advisory, affected package, fixed version).
- Dismiss an alert that is a known-acceptable risk, or reopen a previously dismissed one.

## When NOT to use
- Reviewing **CI / workflow** health → `github-actions-ci-review`.
- Gating one PR's merge → `github-pull-request-review`.
- Filing a remediation ticket from an alert → `github-issue-tracking`.
- Code-scanning or secret-scanning alerts — those are separate GitHub surfaces not covered by
  this tool.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`github-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `GITHUB_TOKEN` | ✅ | Token with `security_events` (or `repo`) scope; org read for `list_org` |
| `GITHUB_URL` | optional | API base (default `https://api.github.com`; set for GHES) |
| `GITHUB_VERIFY` | optional | TLS verification toggle |
| `GITHUB_ALLOW_DESTRUCTIVE` | optional | Set `True` to allow `update` (dismiss/reopen) without per-call `allow_destructive` |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface (below).

## Tools & actions
Prefer the **condensed** tool; it takes `action` + a `params_json` **JSON string**.

| Condensed tool | Actions |
|----------------|---------|
| `github_dependabot` | `list`, `get`, `list_org`, `update` |

### Key parameters
- `list`: `owner` + `repo`, plus optional `state` (`open`/`dismissed`/`fixed`/`auto_dismissed`),
  `severity` (`low`/`medium`/`high`/`critical`), `ecosystem`, `package`, `scope`
  (`development`/`runtime`), `sort`, `direction`, `per_page`.
- `get`: `owner` + `repo` + `alert_number`.
- `list_org`: `org`, plus the same optional filters — one org-wide endpoint.
- `update`: `owner` + `repo` + `alert_number` + `state` (`dismissed`|`open`). To dismiss you
  MUST supply `dismissed_reason` (`fix_started`/`inaccurate`/`no_bandwidth`/`not_used`/
  `tolerable_risk`); `dismissed_comment` is optional. Guarded — pass `allow_destructive=true`
  (or set `GITHUB_ALLOW_DESTRUCTIVE=True`).

## Recipes (`params_json`)
Open critical alerts for a repo:
```json
{"owner":"acme","repo":"api","state":"open","severity":"critical","per_page":50}
```
Every open alert across an org:
```json
{"org":"acme","state":"open","per_page":100}
```
One alert's full detail:
```json
{"owner":"acme","repo":"api","alert_number":42}
```
Dismiss an alert as a tolerable risk (needs `allow_destructive=true`):
```json
{"owner":"acme","repo":"api","alert_number":42,"state":"dismissed","dismissed_reason":"tolerable_risk","dismissed_comment":"Not reachable from a public entrypoint."}
```
Reopen a previously dismissed alert:
```json
{"owner":"acme","repo":"api","alert_number":42,"state":"open"}
```

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- `update` is a **guarded write**: it returns `403` unless `allow_destructive=true` or
  `GITHUB_ALLOW_DESTRUCTIVE=True`. Confirm with the user before dismissing.
- Dismissing **requires** a `dismissed_reason` from GitHub's fixed set; a missing/invalid
  reason is rejected by the API.
- `alert_number` is the per-repo alert number (from the alert URL / list item), not a global id.
- `list` / `list_org` default to a single page — pass `per_page` (max 100) for wider sweeps.

## Related
- **CI health:** `github-actions-ci-review`.
- **Filing remediation issues:** `github-issue-tracking`.
- **KG mapping:** alerts relate to `:Repository` nodes via `github_agent.kg_ingest`.
