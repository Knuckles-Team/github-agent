---
name: github-pull-request-review
description: >-
  Code-review and CI-gated delivery on GitHub via the github-agent MCP server — list,
  read, open, and update pull requests, and check GitHub Actions workflow-run status
  before advising a merge. Use when the agent must review open PRs, open a PR from a
  head branch into a base, update a PR's title/body/state, or verify that CI is green
  on the head ref. Do NOT use for repository/branch structure
  (github-repository-management) or issue triage (github-issue-tracking).
license: MIT
tags: [github, pull-request, code-review, actions, ci, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# GitHub Pull Request Review

Domain-typed access to GitHub **pull requests** and **GitHub Actions** runs via the
github-agent MCP server — the code-review and CI-gate surface. Prefer these condensed
tools over raw REST; they return PR- and run-shaped records.

## When to use
- List / read pull requests, filtered by state, head, or base branch.
- Open a pull request from a `head` branch into a `base` branch.
- Update a PR (title, body, state open/closed, base branch).
- Read GitHub Actions workflow runs and gate a merge recommendation on green CI.
- Rerun, cancel, or trigger a workflow dispatch as part of a delivery decision.

## When NOT to use
- Creating/deleting branches or editing files → `github-repository-management`.
- Triaging issues or cross-repo issue search → `github-issue-tracking`.
- Fetching head-commit check status for the WHOLE fleet in one call → the raw
  `github_graphql` tool with aliased sub-queries (far cheaper than one REST call/repo).

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`github-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `GITHUB_TOKEN` | ✅ | Token with repo scope (and workflow scope to dispatch/rerun) |
| `GITHUB_URL` | optional | API base (default `https://api.github.com`; set for GHES) |
| `GITHUB_VERIFY` | optional | TLS verification toggle |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface (below).

## Tools & actions
Prefer the **condensed** tools; each takes `action` + a `params_json` **JSON string**.

| Condensed tool | Actions |
|----------------|---------|
| `github_pulls` | `list`, `get`, `create`, `update` |
| `github_actions` | `list_workflows`, `list_runs`, `get_run`, `trigger_dispatch`, `rerun`, `cancel`, `delete_run` |

### Key parameters
- `github_pulls`: `owner` + `repo` on every action; `create` needs `title`, `head`,
  `base`; `get`/`update` need `number`.
- `github_actions` `list_runs`: `owner` + `repo` plus optional `status` (e.g.
  `failure`, `success`, `in_progress`), `branch`, `per_page`, `max_pages`.

## Recipes (`params_json`)
List open PRs targeting `main`:
```json
{"owner":"acme","repo":"api","state":"open","base":"main"}
```
Open a pull request:
```json
{"owner":"acme","repo":"api","title":"Add retry backoff","head":"feature/backoff","base":"main","body":"Closes #123"}
```
CI-health sweep — failed runs on the default branch:
```json
{"owner":"acme","repo":"api","status":"failure","branch":"main","per_page":20}
```
Close a PR without merging:
```json
{"owner":"acme","repo":"api","number":42,"state":"closed"}
```

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- `number` is the PR number (`#42`), not the internal `id`.
- Verify the **head** commit's workflow-run `status`/`conclusion` is `success` before
  recommending a merge — a PR being "open" says nothing about CI.
- `github_actions list_runs` responses are slimmed by default (hypermedia dropped,
  `html_url` kept); pass `"slim":false` for full objects.
- `trigger_dispatch`/`rerun` need a token with the **workflow** scope.

## Related
- **KG mapping:** PRs map to `:PullRequest` nodes with `:authoredBy` (→`:Person`) and
  `:belongsToRepository` links via `github_agent.kg_ingest`.
- **Siblings:** `github-repository-management`, `github-issue-tracking`.
