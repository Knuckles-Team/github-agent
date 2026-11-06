---
name: github-pull-request-review
skill_type: skill
description: >-
  Review and merge GitHub pull requests via the github-agent MCP server — by default the PRs assigned to you or where your review is requested, optionally scoped to an org, user, or repo. Read a PR, gate it on green GitHub Actions CI (runs → jobs → job logs), then approve, enable/disable auto-merge, or merge it — write actions are confirmed with the user and CI-gated. Use when the agent must review open PRs, verify CI passes on the head ref, approve, set auto-merge, or merge. Do NOT use to open a PR (github-pull-request-create), triage issues (github-issue-tracking), sweep CI across many repos (github-actions-ci-review), or manage branches/files (github-repository-management).
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Pull Request Review

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Task workflows

### review-a-pull-request

Read a pull request, review it, comment on lines and resolve comments. Contract steps, in order:

1. `github_create_commit_comment`(owner, repo, sha, body; optional params_json)
2. `github_create_pull_request_review`(owner, repo, number; optional event, body, params_json)
3. `github_create_review_comment`(owner, repo, pull_number, body; optional params_json)
4. `github_create_review_comment_reply`(owner, repo, pull_number, comment_id, body)
5. `github_delete_commit_comment`(owner, repo, comment_id; optional confirm) (destructive: pass confirm=true only after the person agrees)
6. `github_delete_review_comment`(owner, repo, comment_id; optional confirm) (destructive: pass confirm=true only after the person agrees)
7. `github_get_commit`(owner, repo, sha)
8. `github_get_commit_comment`(owner, repo, comment_id)
9. `github_get_commits`(owner, repo; optional max_pages, page, per_page, api_parameters, sha, path)
10. `github_get_pull_request`(owner, repo, number)
11. `github_get_pull_requests`(owner, repo; optional max_pages, page, per_page, api_parameters, pull_number, state)
12. `github_get_review_comment`(owner, repo, comment_id)
13. `github_list_commit_comments`(owner, repo, sha; optional params_json)
14. `github_list_repo_commit_comments`(owner, repo; optional params_json)
15. `github_list_repo_review_comments`(owner, repo; optional params_json)
16. `github_list_review_comments`(owner, repo, pull_number; optional params_json)
17. `github_update_commit_comment`(owner, repo, comment_id, body)
18. `github_update_review_comment`(owner, repo, comment_id, body)

### land-a-pull-request

Update, merge or auto-merge a pull request and delete its branch. Contract steps, in order:

1. `github_delete_branch`(owner, repo, branch; optional confirm) (destructive: pass confirm=true only after the person agrees)
2. `github_merge_pull_request`(owner, repo, number; optional merge_method, commit_title, commit_message, sha)
3. `github_pulls_disable_auto_merge`(owner, repo, number)
4. `github_pulls_enable_auto_merge`(owner, repo, number; optional merge_method)
5. `github_update_pull_request`(owner, repo, number; optional params_json)
