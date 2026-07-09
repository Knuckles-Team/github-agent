---
name: github-pull-request-review
skill_type: skill
description: >-
  Review and merge GitHub pull requests via the github-agent MCP server — by default the
  PRs assigned to you or where your review is requested, optionally scoped to an org, user,
  or repo. Read a PR, gate it on green GitHub Actions CI (runs → jobs → job logs), then
  approve, enable/disable auto-merge, or merge it — write actions are confirmed with the
  user and CI-gated. Use when the agent must review open PRs, verify CI passes on the head
  ref, approve, set auto-merge, or merge. Do NOT use to open a PR (github-pull-request-create),
  triage issues (github-issue-tracking), sweep CI across many repos (github-actions-ci-review),
  or manage branches/files (github-repository-management).
license: MIT
tags: [github, pull-request, code-review, approve, merge, auto-merge, actions, ci, mcp]
metadata:
  author: Genius
  version: '0.2.0'
---
# GitHub Pull Request Review

Domain-typed access to GitHub **pull requests** and **GitHub Actions** via the github-agent
MCP server — the review-to-merge surface. Default focus: the PRs that are *yours to act on*
(assigned to you or awaiting your review); widen to an org/user/repo on request. Prefer these
condensed tools over raw REST.

## When to use
- Review the PRs assigned to you or where your review is requested (the default).
- Scope PR review to a specific `org`, `user`, or `repo`.
- Read a PR and gate a merge on green CI — resolve the head ref's workflow runs, jobs, and
  job logs before advising.
- **Approve** a PR, **request reviewers**, **enable/disable auto-merge**, or **merge** it.

## When NOT to use
- Opening a new PR → `github-pull-request-create`.
- Issue triage / issue search → `github-issue-tracking`.
- A CI-health sweep across a whole org/user's repos → `github-actions-ci-review`.
- Branch/file/collaborator structure → `github-repository-management`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`github-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `GITHUB_TOKEN` | ✅ | Token with `repo` scope; `workflow` scope to rerun/dispatch |
| `GITHUB_URL` | optional | API base (default `https://api.github.com`; set for GHES) |
| `GITHUB_VERIFY` | optional | TLS verification toggle |
| `GITHUB_ALLOW_DESTRUCTIVE` | optional | Env fallback that unlocks the guarded writes `merge` / `enable_auto_merge` |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface (below).

## Tools & actions
Prefer the **condensed** tools; each takes `action` + a `params_json` **JSON string**.

| Condensed tool | Actions |
|----------------|---------|
| `github_search` | `issues` (find PRs by qualifier — the default entry point) |
| `github_pulls` | `list`, `get`, `create`, `update`, `approve`, `request_reviewers`, `merge`, `enable_auto_merge`, `disable_auto_merge` |
| `github_actions` | `list_runs`, `get_run`, `list_jobs`, `job_logs`, `rerun`, `cancel` |

### Key parameters
- `github_search issues`: `q` — a GitHub search query. PRs are matched with `is:pr`; the
  default reviewer set is `is:pr is:open review-requested:@me` and `is:pr is:open assignee:@me`.
  Scope by adding `org:<org>`, `user:<user>`, or `repo:<owner>/<repo>` to `q`.
- `github_pulls`: `owner` + `repo` on every action; `get`/`update`/`approve`/`merge`/
  `request_reviewers` need `number`. `merge` takes `merge_method` (merge/squash/rebase) and
  optional `commit_title`/`commit_message`/`sha`. `enable_auto_merge` takes `merge_method`
  (MERGE/SQUASH/REBASE) and accepts either `owner`+`repo`+`number` or a `pull_request_id` node id.
- `github_actions`: `list_runs` filters by `branch`/`head_sha`/`status`; `list_jobs` needs
  `run_id`; `job_logs` needs `job_id` and returns the decoded plaintext log.
- **Guarded writes:** `merge` and `enable_auto_merge` are blocked unless the call passes
  `allow_destructive=true` (top-level tool arg) or `GITHUB_ALLOW_DESTRUCTIVE=True` is set.

## Recipes (`params_json`)
Default — PRs awaiting your review (open, review requested from you):
```json
{"q":"is:pr is:open review-requested:@me","sort":"updated","order":"desc"}
```
PRs assigned to you across one org:
```json
{"q":"is:pr is:open assignee:@me org:acme"}
```
Read a PR before deciding:
```json
{"owner":"acme","repo":"api","number":42}
```
Gate on CI — the head ref's latest runs, then its jobs, then a failing job's log:
```json
{"owner":"acme","repo":"api","branch":"feature/backoff","per_page":5}
```
```json
{"owner":"acme","repo":"api","run_id":80512345}
```
```json
{"owner":"acme","repo":"api","job_id":22110987}
```
Approve a PR:
```json
{"owner":"acme","repo":"api","number":42,"event":"APPROVE","body":"CI green, LGTM."}
```
Enable auto-merge (squash) — guarded, pass `allow_destructive=true` on the tool call:
```json
{"owner":"acme","repo":"api","number":42,"merge_method":"SQUASH"}
```
Merge now (squash) — guarded:
```json
{"owner":"acme","repo":"api","number":42,"merge_method":"squash","commit_title":"Add retry backoff (#42)"}
```

## Guarded write actions
`approve`, `request_reviewers`, `merge`, `enable_auto_merge`, and `disable_auto_merge` mutate
the PR. Before calling any of them:
1. **Confirm CI is green** — the head ref's workflow runs are `conclusion=success` and no
   required job log shows a failure. Recommend `merge`/`enable_auto_merge` only when CI passes
   (prefer `enable_auto_merge` when checks are still running).
2. **Present the exact action to the user** — repo, PR number/title, and the operation
   (approve / auto-merge with method / immediate merge) — and get explicit confirmation.
3. Only then call the tool; `merge` and `enable_auto_merge` additionally require
   `allow_destructive=true`.

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- `github_search issues` returns both issues and PRs; keep `is:pr` in `q` to stay on PRs.
- `number` is the PR number (`#42`), not the internal `id`.
- A PR being "open" says nothing about CI — always verify the **head** run
  `conclusion=success` before recommending a merge.
- `enable_auto_merge` uses GraphQL under the hood; it needs auto-merge enabled on the repo
  and a token with write access. Provide `owner`+`repo`+`number` and it resolves the node id.
- `job_logs` returns the full decoded text — request one failing job at a time to stay small.

## Related
- **KG mapping:** PRs map to `:PullRequest` nodes with `:authoredBy` (→`:Person`) and
  `:belongsToRepository` links via `github_agent.kg_ingest`.
- **Siblings:** `github-pull-request-create`, `github-actions-ci-review`,
  `github-issue-tracking`.
