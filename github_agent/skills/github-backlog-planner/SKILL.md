---
name: github-backlog-planner
skill_type: skill
description: >-
  Sweeps every open issue and pull request across one or more GitHub accounts (a user and/or an organization, all repositories), deeply verifies which items are already addressed, and emits a prioritized Markdown remediation plan. Use when asked to "go through all issues and PRs", triage a whole backlog, plan what to work on next across repos, or audit open work for a user/org. Defaults to read-only analysis; write actions (close, comment, open tracking issue) are opt-in and run only on explicit confirmation. Do NOT use for a single PR's review comments (use github-tools) or interactive one-tracker triage (use issue-triage).
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Backlog Planner

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Task workflows

### work-issues

Read, open, change and comment on issues. Contract steps, in order:

1. `github_create_issue`(owner, repo, title; optional params_json)
2. `github_create_issue_comment`(owner, repo, issue_number, body)
3. `github_delete_issue_comment`(owner, repo, comment_id; optional confirm) (destructive: pass confirm=true only after the person agrees)
4. `github_get_issue`(owner, repo, number)
5. `github_get_issue_comment`(owner, repo, comment_id)
6. `github_get_issues`(owner, repo; optional max_pages, page, per_page, api_parameters, issue_number, state)
7. `github_list_issue_comments`(owner, repo, issue_number; optional params_json)
8. `github_list_repo_issue_comments`(owner, repo; optional params_json)
9. `github_search_issues`(q; optional max_pages, page, per_page, api_parameters, sort, order)
10. `github_update_issue`(owner, repo, number; optional params_json)
11. `github_update_issue_comment`(owner, repo, comment_id, body)

### find-repositories

Find repositories and read their details. Contract steps, in order:

1. `github_get_org_repos`(org; optional max_pages, page, per_page, api_parameters, type)
2. `github_get_organization`(org)
3. `github_get_repositories`(; optional max_pages, page, per_page, api_parameters, owner, repo)
4. `github_get_repository`(owner, repo)
5. `github_list_organizations`(; optional max_pages, page, per_page, api_parameters, scope, since)
6. `github_search_repositories`(q; optional max_pages, page, per_page, api_parameters, sort, order)
