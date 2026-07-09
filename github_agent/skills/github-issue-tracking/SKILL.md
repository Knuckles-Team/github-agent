---
name: github-issue-tracking
skill_type: skill
description: >-
  Issue triage and search on GitHub via the github-agent MCP server — list, read, open,
  and update issues (repo-scoped or org-wide in one search call), and run repository /
  issue / code searches. Use when the agent must triage open issues by label or
  assignee, open or update an issue, find every open issue across an org, or search
  repositories/code. Do NOT use for pull-request review
  (github-pull-request-review), opening a PR (github-pull-request-create), CI/Actions review
  (github-actions-ci-review), or repository/branch structure (github-repository-management).
license: MIT
tags: [github, issues, search, triage, organization, mcp]
metadata:
  author: Genius
  version: '0.2.0'
---
# GitHub Issue Tracking

Domain-typed access to GitHub **issues** and **search** via the github-agent MCP server
— the triage and discovery surface. Prefer these condensed tools over raw REST; they
return issue- and search-shaped records.

## When to use
- Review / triage issues for a **specific repository** (project), filtered by state, labels,
  or assignee.
- Review issues affecting a whole **organization** — every open issue across its repos in ONE
  `/search/issues` call.
- Open a new issue or update one (title, body, state, labels, assignees).
- Search repositories, issues, or code with GitHub query qualifiers.

## When NOT to use
- Pull-request review or CI gating → `github-pull-request-review`.
- Creating branches, committing files, or managing collaborators →
  `github-repository-management`.
- A returned issue item carrying a `pull_request` field is actually a PR — handle it
  with `github-pull-request-review`, not here.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`github-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `GITHUB_TOKEN` | ✅ | Token with repo scope |
| `GITHUB_URL` | optional | API base (default `https://api.github.com`; set for GHES) |
| `GITHUB_VERIFY` | optional | TLS verification toggle |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface (below).

## Tools & actions
Prefer the **condensed** tools; each takes `action` + a `params_json` **JSON string**.

| Condensed tool | Actions |
|----------------|---------|
| `github_issues` | `list`, `get`, `create`, `update` |
| `github_search` | `repositories`, `issues`, `code` |

### Key parameters
- `github_issues list`: EITHER repo-scoped (`owner` + `repo`) OR org-wide (`org` alone,
  which fans out via one search call). Filters: `state` (open/closed/all), `labels`
  (comma-separated), `assignee`, `since`, `per_page`, `max_pages`.
- `github_issues create`: `owner` + `repo` + `title` (optional `body`, `labels`,
  `assignees`); `get`/`update` need `number`.
- `github_search`: `q` (GitHub query string) plus optional `sort`, `order`, `per_page`.

## Recipes (`params_json`)
Triage open `bug` issues assigned to a user in a repo:
```json
{"owner":"acme","repo":"api","state":"open","labels":"bug","assignee":"octocat"}
```
Every open issue across an org in ONE call:
```json
{"org":"acme","state":"open"}
```
Open an issue with labels:
```json
{"owner":"acme","repo":"api","title":"Flaky retry test","body":"Fails ~5% on CI","labels":["bug","ci"]}
```
Search code for a symbol in an org:
```json
{"q":"org:acme retry_backoff in:file language:python"}
```

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- `number` is the issue number (`#17`), not the internal `id`.
- The org-wide `list` path uses `/search/issues` with `is:issue`, so it **excludes**
  PRs; the repo-scoped path may include PRs (filter items with a `pull_request` field).
- Search endpoints are rate-limited more tightly than the core API — prefer a specific
  `q` and a sane `per_page`.

## Related
- **KG mapping:** issues map to `:Issue` nodes with `:authoredBy` (→`:Person`) and
  `:belongsToRepository` links via `github_agent.kg_ingest` (PR-shaped items skipped).
- **Siblings:** `github-repository-management`, `github-pull-request-review`.
