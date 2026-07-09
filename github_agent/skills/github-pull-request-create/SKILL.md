---
name: github-pull-request-create
description: >-
  Open a new GitHub pull request against a repository via the github-agent MCP server —
  from a head branch into a base branch, with title, body, draft flag, and requested
  reviewers. Use when the agent must create/open a PR for a project. Do NOT use to review,
  approve, or merge an existing PR (github-pull-request-review), to triage issues
  (github-issue-tracking), or to create branches/commit files first
  (github-repository-management).
license: MIT
tags: [github, pull-request, create, reviewers, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# GitHub Pull Request Create

Domain-typed creation of a GitHub **pull request** against a repository via the github-agent
MCP server. This skill opens the PR and optionally requests reviewers; reviewing/approving/
merging it is `github-pull-request-review`.

## When to use
- Open a PR from a `head` branch into a `base` branch on a repo.
- Open it as a **draft**, set the title/body, and **request reviewers** (users or teams).

## When NOT to use
- Reviewing, approving, auto-merging, or merging an existing PR → `github-pull-request-review`.
- Creating the head branch or committing the files the PR will contain →
  `github-repository-management`.
- Filing an issue → `github-issue-tracking`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`github-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `GITHUB_TOKEN` | ✅ | Token with `repo` scope (write) |
| `GITHUB_URL` | optional | API base (default `https://api.github.com`; set for GHES) |
| `GITHUB_VERIFY` | optional | TLS verification toggle |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface (below).

## Tools & actions
Prefer the **condensed** tool; it takes `action` + a `params_json` **JSON string**.

| Condensed tool | Actions |
|----------------|---------|
| `github_pulls` | `create`, `request_reviewers` |

### Key parameters
- `github_pulls create`: `owner` + `repo` + `title` + `head` (source branch) + `base` (target
  branch); optional `body`, `draft` (bool), `maintainer_can_modify`. Cross-fork head uses
  `head` as `owner:branch`.
- `github_pulls request_reviewers`: `owner` + `repo` + `number` (the new PR's number) plus
  `reviewers` (list of usernames) and/or `team_reviewers` (list of team slugs).

## Recipes (`params_json`)
Open a pull request:
```json
{"owner":"acme","repo":"api","title":"Add retry backoff","head":"feature/backoff","base":"main","body":"Closes #123"}
```
Open a draft PR:
```json
{"owner":"acme","repo":"api","title":"WIP: caching layer","head":"feature/cache","base":"main","draft":true}
```
Request reviewers on the PR just created (use its returned `number`):
```json
{"owner":"acme","repo":"api","number":57,"reviewers":["octocat","hubot"],"team_reviewers":["platform"]}
```

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- `head` and `base` are branch names, not SHAs; the head branch must already exist and have
  commits ahead of `base` or GitHub rejects the PR.
- Requesting a reviewer who lacks repo access, or requesting yourself, returns a 422.
- To open across a fork, set `head` to `fork_owner:branch`.

## Related
- **Next step:** review/approve/merge the PR with `github-pull-request-review`.
- **KG mapping:** the new PR maps to a `:PullRequest` node via `github_agent.kg_ingest`.
