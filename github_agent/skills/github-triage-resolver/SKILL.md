---
name: github-triage-resolver
skill_type: skill
description: >-
  Triages open PRs and issues across a GitHub user or org, decides resolve-or-skip from evidence, and closes or merges safe items with a comment. Use to actually clear a backlog or auto-merge safe dependency bumps. Read-only by default, writes need confirmation. Do NOT use for report-only sweeps or CI triage.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Triage Resolver

Scope: Triages open pull requests and issues across a GitHub user/org (all repos), verifies whether each is still real/relevant, decides resolve-or-skip, and — only when deemed safe — closes or merges it with an explanatory comment. For each item it gathers evidence (is the issue already fixed? is the PR mergeable with green checks?), classifies it safe_merge / safe_close / skip via deterministic rules, and acts behind a dry-run + explicit-confirmation gate. Use when asked to go through open PRs/issues and actually resolve them, auto-merge safe dependency bumps, close already-fixed issues or abandoned PRs with a comment, or clear a backlog with action (not just a report). Read-only by default; every write needs confirmation. Do NOT use for read-only reporting (use github-pr-review-sweep or github-backlog-planner), for CI triage (github-ci-failure-sweep), or for GitLab.

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Steps

1. Collect open PRs and issues with `github_get_pull_requests`(owner, repo) and `github_get_issues`(owner, repo, state="open"). Verify each item has a number and updated date.
2. Gather evidence per item with `github_get_commits`(owner, repo) and CI from `github_get_workflow_runs`(owner, repo). Verify each item has evidence recorded.
3. Classify with `python scripts/classify_safe.py items.json`, using `references/resolution-rules.md`. Verify each item is safe_merge, safe_close or skip.
4. Show the dry-run plan with `python scripts/classify_safe.py items.json --format md` and wait for explicit confirmation. Verify the person approved each write item.
5. Apply an approved action with `python scripts/gh_write.py comment owner/repo 12 --body text --confirm` or the merge subcommand. Verify with `github_get_issue` or `github_get_pull_request` that the state changed.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| HTTP 401 or 403 from a github_* tool | `GITHUB_TOKEN` is missing, expired or lacks the scope for this owner | Set a valid `GITHUB_TOKEN` with the needed scope in the server environment, and call `github_get_repository`(owner, repo) to verify it returns the repository |
| gh_write refuses to run | The --confirm flag is missing | Pass --confirm only after the person approved that item, and verify the state afterwards |
| Item classified skip | Evidence is missing or checks are not green | Leave the item open and report the missing evidence; do not force a merge |

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
