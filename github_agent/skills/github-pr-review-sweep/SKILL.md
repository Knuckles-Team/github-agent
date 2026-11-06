---
name: github-pr-review-sweep
skill_type: skill
description: >-
  Sweeps open pull requests across one or more GitHub accounts (a user and/or an organization, all repositories) and produces a per-PR review summary: for each open PR it gathers the diff size, mergeable state, required-review/check status, conflicts, and staleness, then classifies it as ready-to-merge, changes-requested, failing-checks, conflicts, or draft, with a recommended next action. Use when asked to review all open PRs across the org, audit pull-request health for a user/org, see which PRs are ready to merge or blocked, or triage review workload across repos. Defaults to read-only analysis; leaving a review, commenting, or merging is opt-in and runs only on explicit confirmation. Do NOT use for a single PR's checks (use github-tools), for issue/PR backlog planning (use github-backlog-planner), for CI failure triage (use github-ci-failure-sweep), or for GitLab merge requests.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Pr Review Sweep

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
