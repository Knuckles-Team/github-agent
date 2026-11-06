---
name: github-org-remediation-loop
skill_type: skill
description: >-
  Sweeps organization issues, failing CI, and incoming pull requests; validates each item; creates an SDD spec; and risk-classifies remediation. Low-risk work may be implemented, merged, and closed; elevated-risk work is implemented on a branch for review. Use for backlog remediation, CI fixes, or incoming-PR triage. It always performs a read-only sweep, specification, and risk classification first; every write is autonomy-gated and defaults to dry-run. Run through github-agent with graph_orchestrate. For a single PR review or a report-only sweep, use the focused review or backlog skills instead.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Org Remediation Loop

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Task workflows

### administer-organization

Manage organization membership, teams and settings. Contract steps, in order:

1. `github_create_organization`(login, admin; optional profile_name)
2. `github_delete_organization`(org; optional confirm) (destructive: pass confirm=true only after the person agrees)
3. `github_get_org_members`(org; optional max_pages, page, per_page, api_parameters, role)
4. `github_get_org_teams`(org)
5. `github_get_organization_membership`(org, username)
6. `github_remove_organization_member`(org, username; optional confirm) (destructive: pass confirm=true only after the person agrees)
7. `github_set_organization_membership`(org, username; optional role)
8. `github_update_organization`(org; optional params_json)

### review-dependabot-alerts

Read and dismiss Dependabot alerts of a repository or organization. Contract steps, in order:

1. `github_get_dependabot_alert`(owner, repo, alert_number)
2. `github_get_dependabot_alerts`(owner, repo; optional params_json)
3. `github_get_org_dependabot_alerts`(org; optional params_json)
4. `github_update_dependabot_alert`(owner, repo, alert_number, state; optional dismissed_reason, dismissed_comment)

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

### land-a-pull-request

Update, merge or auto-merge a pull request and delete its branch. Contract steps, in order:

1. `github_delete_branch`(owner, repo, branch; optional confirm) (destructive: pass confirm=true only after the person agrees)
2. `github_merge_pull_request`(owner, repo, number; optional merge_method, commit_title, commit_message, sha)
3. `github_pulls_disable_auto_merge`(owner, repo, number)
4. `github_pulls_enable_auto_merge`(owner, repo, number; optional merge_method)
5. `github_update_pull_request`(owner, repo, number; optional params_json)
