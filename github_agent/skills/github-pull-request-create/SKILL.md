---
name: github-pull-request-create
skill_type: skill
description: >-
  Open a new GitHub pull request against a repository via the github-agent MCP server — from a head branch into a base branch, with title, body, draft flag, and requested reviewers. Use when the agent must create/open a PR for a project. Do NOT use to review, approve, or merge an existing PR (github-pull-request-review), to triage issues (github-issue-tracking), or to create branches/commit files first (github-repository-management).
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Pull Request Create

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

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
