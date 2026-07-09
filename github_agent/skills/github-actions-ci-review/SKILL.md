---
name: github-actions-ci-review
description: >-
  Review GitHub Actions CI results across an organization, a user, or a single repository via
  the github-agent MCP server — enumerate repos, list workflow runs (filterable to failures),
  drill into a run's jobs, and read a job's logs. Use when the agent must check CI health for
  an org/user/repo, find red pipelines, or read why a run/job failed. Do NOT use to gate a
  single PR's merge on CI (github-pull-request-review), to trigger a deploy workflow
  (github-repository-management), or to manage issues (github-issue-tracking).
license: MIT
tags: [github, actions, ci, workflows, pipelines, jobs, logs, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# GitHub Actions CI Review

Domain-typed review of **GitHub Actions** CI results via the github-agent MCP server, scoped to
an **organization**, a **user**, or a **single repository**. Surfaces pipeline (workflow-run)
results, the jobs within a run, and per-job logs. Prefer these condensed tools over raw REST.

## When to use
- Review CI results for every repo in an **org** (`github_orgs repos`) or a **user**
  (`github_repos list`), or for one **repo** directly.
- List workflow runs, filtered to `status=failure` for a red-CI sweep.
- Drill into a run's **jobs** and read a failing **job's logs**.

## When NOT to use
- Gating one PR's merge on its head-ref CI → `github-pull-request-review` (it already reads
  runs/jobs/logs for the PR).
- Triggering / re-running / cancelling a workflow as a delivery action →
  `github-repository-management` or the `github_actions` `trigger_dispatch`/`rerun`/`cancel`
  actions directly.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`github-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `GITHUB_TOKEN` | ✅ | Token with `repo` + `actions:read` scope |
| `GITHUB_URL` | optional | API base (default `https://api.github.com`; set for GHES) |
| `GITHUB_VERIFY` | optional | TLS verification toggle |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface (below).

## Tools & actions
Prefer the **condensed** tools; each takes `action` + a `params_json` **JSON string**.

| Condensed tool | Actions |
|----------------|---------|
| `github_orgs` | `repos` (enumerate an org's repos) |
| `github_repos` | `list` (enumerate a user's / your repos) |
| `github_actions` | `list_workflows`, `list_runs`, `get_run`, `list_jobs`, `job_logs` |

### Key parameters
- Scope resolution: an **org** → `github_orgs repos` with `org`; a **user** → `github_repos
  list`; a **repo** → skip enumeration and go straight to `github_actions`.
- `github_actions list_runs`: `owner` + `repo`, plus optional `status` (`failure`, `success`,
  `in_progress`, `completed`), `branch`, `per_page`, `max_pages`. Responses are slimmed by
  default (`slim=false` for full objects).
- `github_actions list_jobs`: `owner` + `repo` + `run_id` (optional `filter`: latest|all).
- `github_actions job_logs`: `owner` + `repo` + `job_id` → decoded plaintext log.

## Recipes (`params_json`)
Enumerate an org's repos to sweep:
```json
{"org":"acme","per_page":100}
```
Failed runs on the default branch of a repo:
```json
{"owner":"acme","repo":"api","status":"failure","branch":"main","per_page":20}
```
Jobs of a specific run:
```json
{"owner":"acme","repo":"api","run_id":80512345}
```
Read a failing job's log:
```json
{"owner":"acme","repo":"api","job_id":22110987}
```

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- Judge a run by its `conclusion` (`success`/`failure`/`cancelled`), not its `status`
  (`queued`/`in_progress`/`completed`) — `completed` alone does not mean it passed.
- An org sweep fans out one `list_runs` call per repo; cap `per_page` and prefer
  `status=failure` to keep it cheap. There is no single org-wide runs endpoint.
- `job_logs` returns the full decoded text — pull one failing job at a time.

## Related
- **Per-PR CI gating:** `github-pull-request-review`.
- **KG mapping:** runs relate to `:Repository` nodes via `github_agent.kg_ingest`.
