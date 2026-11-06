---
name: github-dependabot-review
skill_type: skill
description: >-
  Review GitHub Dependabot vulnerability alerts for a single repository or an entire organization via the github-agent MCP server — list alerts (filter by severity, state, ecosystem, package, scope), drill into a single alert, and dismiss or reopen an alert with explicit confirmation. Use when the agent must audit dependency vulnerabilities, triage security alerts across an org/repo, or dismiss a known-acceptable alert. Do NOT use for code scanning / secret scanning alerts (not covered here), to gate a PR's merge on CI (github-actions-ci-review), or to open remediation issues (github-issue-tracking).
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Dependabot Review

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Task workflows

### review-dependabot-alerts

Read and dismiss Dependabot alerts of a repository or organization. Contract steps, in order:

1. `github_get_dependabot_alert`(owner, repo, alert_number)
2. `github_get_dependabot_alerts`(owner, repo; optional params_json)
3. `github_get_org_dependabot_alerts`(org; optional params_json)
4. `github_update_dependabot_alert`(owner, repo, alert_number, state; optional dismissed_reason, dismissed_comment)
