---
name: github-issue-tracking
skill_type: skill
description: >-
  Issue triage and search on GitHub via the github-agent MCP server — list, read, open, and update issues (repo-scoped or org-wide in one search call), and run repository / issue / code searches. Use when the agent must triage open issues by label or assignee, open or update an issue, find every open issue across an org, or search repositories/code. Do NOT use for pull-request review (github-pull-request-review), opening a PR (github-pull-request-create), CI/Actions review (github-actions-ci-review), or repository/branch structure (github-repository-management).
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Issue Tracking

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
