---
name: github-issue-tracking
skill_type: skill
description: >-
  Issue triage and search on GitHub via the github-agent MCP server — list, read, open, and update issues (repo-scoped or org-wide in one search call), and run repository / issue / code searches. Use when the agent must triage open issues by label or assignee, open or update an issue, find every open issue across an org, or search repositories/code.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Issue Tracking

Scope: Issue triage and search on GitHub via the github-agent MCP server — list, read, open, and update issues (repo-scoped or org-wide in one search call), and run repository / issue / code searches. Use when the agent must triage open issues by label or assignee, open or update an issue, find every open issue across an org, or search repositories/code. Do NOT use for pull-request review (github-pull-request-review), opening a PR (github-pull-request-create), CI/Actions review (github-actions-ci-review), or repository/branch structure (github-repository-management).

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Steps

1. Resolve owner and repo, or an org for an org-wide search. Verify with `github_get_repository`(owner, repo).
2. List or search issues with `github_get_issues`(owner, repo, state) or `github_search_issues`(q). Verify the results match the label or assignee asked for.
3. Read the chosen issue with `github_get_issue`(owner, repo, number) and its comments with `github_list_issue_comments`(owner, repo, issue_number). Verify the current state and body.
4. Open or change an issue only when asked, with `github_create_issue`(owner, repo, title) or `github_update_issue`(owner, repo, number). Verify by reading it back with `github_get_issue`.
5. Add a comment with `github_create_issue_comment`(owner, repo, issue_number, body). Verify the comment appears in `github_list_issue_comments`.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| HTTP 401 or 403 from a github_* tool | `GITHUB_TOKEN` is missing, expired or lacks the scope for this owner | Set a valid `GITHUB_TOKEN` with the needed scope in the server environment, and call `github_get_repository`(owner, repo) to verify it returns the repository |
| HTTP 404 for a repository, run or number | owner, repo or the numeric id is wrong, or the token cannot see a private repository | Re-resolve the names with `github_search_repositories`(q) and use the exact owner/repo and id it returns |
| Search returns nothing | The query qualifiers are too narrow or mis-spelled | Simplify the `q` string to `repo:owner/repo is:issue is:open` and add qualifiers one at a time |

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
