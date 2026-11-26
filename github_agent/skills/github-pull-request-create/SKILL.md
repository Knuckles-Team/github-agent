---
name: github-pull-request-create
skill_type: skill
description: >-
  Open a new GitHub pull request against a repository via the github-agent MCP server — from a head branch into a base branch, with title, body, draft flag, and requested reviewers. Use when the agent must create/open a PR for a project.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Pull Request Create

Scope: Open a new GitHub pull request against a repository via the github-agent MCP server — from a head branch into a base branch, with title, body, draft flag, and requested reviewers. Use when the agent must create/open a PR for a project. Do NOT use to review, approve, or merge an existing PR (github-pull-request-review), to triage issues (github-issue-tracking), or to create branches/commit files first (github-repository-management).

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Steps

1. Confirm the head and base branches with `github_get_branches`(owner, repo). Verify both names exist.
2. Create the head branch if needed with `github_create_branch`(owner, repo, branch, ref) and commit files with `github_create_content`(owner, repo, path, message, content). Verify with `github_get_branch`(owner, repo, branch).
3. Open the PR with `github_create_pull_request`(owner, repo, title, head, base). Verify the response returns a PR number and URL.
4. Request reviewers with `github_request_reviewers`(owner, repo, number, reviewers). Verify with `github_get_pull_request`(owner, repo, number) that they are listed.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| HTTP 401 or 403 from a github_* tool | `GITHUB_TOKEN` is missing, expired or lacks the scope for this owner | Set a valid `GITHUB_TOKEN` with the needed scope in the server environment, and call `github_get_repository`(owner, repo) to verify it returns the repository |
| 422 on create | No commits differ between head and base, or a PR already exists | Compare with `github_get_commits`(owner, repo) and open the PR only when the head has new commits |
| Reviewer request rejected | The reviewer lacks access to the repository | Add them with `github_add_collaborator`(owner, repo, username) or choose another reviewer |

## Task workflows

### create-a-pull-request

Branch, commit content and open a pull request. Contract steps, in order:

1. `github_create_branch`(owner, repo, branch, ref)
2. `github_create_content`(owner, repo, path, message, content; optional params_json)
3. `github_create_pull_request`(owner, repo, title, head, base; optional params_json)
4. `github_get_branch`(owner, repo, branch)
5. `github_get_branches`(owner, repo; optional max_pages, page, per_page, api_parameters, branch)
6. `github_get_contents`(owner, repo, path; optional max_pages, page, per_page, api_parameters, ref)
7. `github_request_reviewers`(owner, repo, number; optional reviewers, team_reviewers)
8. `github_update_content`(owner, repo, path, message, content, sha; optional params_json)
