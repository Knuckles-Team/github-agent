---
name: github-repository-management
skill_type: skill
description: >-
  Repository lifecycle and structure operations on GitHub via the github-agent MCP
  server — list/read/create/update/delete repositories, manage branches and branch
  protection, read and write file contents, manage collaborators, and enable/configure
  GitHub Pages. Use when the agent must inspect or provision a repo, set up branch
  protection, commit a file, add a collaborator, or turn on Pages. Do NOT use for
  pull-request review (github-pull-request-review), issue triage
  (github-issue-tracking), or raw GraphQL fan-out (use the github_graphql tool).
license: MIT
tags: [github, repository, branches, contents, pages, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# GitHub Repository Management

Domain-typed access to GitHub **repositories** and their structure — repos, branches,
file contents, collaborators, and GitHub Pages — via the github-agent MCP server. Prefer
these condensed tools over raw REST calls; they carry GitHub's field conventions and
return typed records.

## When to use
- List / read / create / update / delete repositories (for a user or an org).
- Manage branches: list, get, create from a ref, delete, and get/set/delete branch
  protection.
- Read a file or directory, or create/update/delete a file (Contents API).
- Manage repository collaborators (list / add with a permission / remove).
- Enable, configure, or rebuild a GitHub Pages site.

## When NOT to use
- Reviewing or merging pull requests → `github-pull-request-review`.
- Triaging issues or searching across repos → `github-issue-tracking`.
- One request fanning out across many repos (e.g. fleet-wide CI status) → the raw
  `github_graphql` tool with aliased sub-queries.
- Pushing repo state into the knowledge graph → the `github_ingest_repos` misc tool
  (ingestion plumbing, not an operational action).

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`github-agent`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `GITHUB_TOKEN` | ✅ | Personal-access / app token with repo scope |
| `GITHUB_URL` | optional | API base (default `https://api.github.com`; set for GHES) |
| `GITHUB_VERIFY` | optional | TLS verification toggle |
| `GITHUB_ALLOW_DESTRUCTIVE` | optional | Must be true (or pass `allow_destructive=true`) for `pages_delete` |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed surface (used
below) vs. the one-to-one verbose tools.

## Tools & actions
Prefer the **condensed** tools; each takes `action` + a `params_json` **JSON string**
whose keys are passed to the client method.

| Condensed tool | Actions |
|----------------|---------|
| `github_repos` | `list`, `get`, `create`, `update`, `delete`, `pages_get`, `pages_create`, `pages_update`, `pages_delete`, `pages_builds`, `pages_request_build` |
| `github_branches` | `list`, `get`, `create`, `delete`, `get_protection`, `update_protection`, `delete_protection` |
| `github_contents` | `get`, `create`, `update`, `delete` |
| `github_collaborators` | `list`, `add`, `remove` |

### Key parameters
- `owner` + `repo` scope almost every action; `create` needs `name`.
- Contents `create`/`update` need `path`, `message`, base64 `content`; `update`/`delete`
  also need the current blob `sha`.
- `pages_create` takes `build_type` (`workflow` default, or `legacy` + `source`).

## Recipes (`params_json`)
List the authenticated user's private repositories:
```json
{"visibility":"private","affiliation":"owner"}
```
Create a repo, auto-initialized:
```json
{"name":"new-service","description":"service scaffold","private":true,"auto_init":true}
```
Require a green check + one review on `main` (branch protection):
```json
{"owner":"acme","repo":"api","branch":"main","protection_config":{"required_status_checks":{"strict":true,"contexts":["ci"]},"enforce_admins":true,"required_pull_request_reviews":{"required_approving_review_count":1},"restrictions":null}}
```
Add a collaborator with write access:
```json
{"owner":"acme","repo":"api","username":"octocat","permission":"push"}
```

## Gotchas
- `params_json` is a **string** of JSON, not an object — serialize it.
- `list` responses are **slimmed** by default (`*_url`/`_links` hypermedia dropped,
  `html_url` kept); pass `"slim":false` for the full objects.
- Contents `update`/`delete` fail without the current `sha` — `get` the file first.
- `pages_delete` is destructive and blocked unless `allow_destructive=true` (or
  `GITHUB_ALLOW_DESTRUCTIVE=True`).
- After a first `pages_create`, the initial build can be skipped — call
  `pages_request_build` to force it.

## Related
- **KG ingestion:** `github_ingest_repos` pushes listed repos into the knowledge graph
  as typed `:Repository` nodes (with owner `:Organization`/`:Person` links).
- **Siblings:** `github-pull-request-review`, `github-issue-tracking`.
