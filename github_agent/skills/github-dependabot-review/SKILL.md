---
name: github-dependabot-review
skill_type: skill
description: >-
  Review GitHub Dependabot vulnerability alerts for a repository or organization via the github-agent MCP server: list and filter alerts, read one, and dismiss or reopen with confirmation. Use to audit dependency vulnerabilities or triage security alerts. Do NOT use for code or secret scanning alerts, PR CI gating, or opening remediation issues.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Dependabot Review

Scope: Review GitHub Dependabot vulnerability alerts for a single repository or an entire organization via the github-agent MCP server — list alerts (filter by severity, state, ecosystem, package, scope), drill into a single alert, and dismiss or reopen an alert with explicit confirmation. Use when the agent must audit dependency vulnerabilities, triage security alerts across an org/repo, or dismiss a known-acceptable alert. Do NOT use for code scanning / secret scanning alerts (not covered here), to gate a PR's merge on CI (github-actions-ci-review), or to open remediation issues (github-issue-tracking).

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Steps

1. Choose the scope: one repository or a whole organization. Verify with `github_get_repository`(owner, repo) or `github_get_organization`(org) that it exists.
2. List alerts with `github_get_dependabot_alerts`(owner, repo) or `github_get_org_dependabot_alerts`(org). Verify each alert shows severity, package and state.
3. Open one alert with `github_get_dependabot_alert`(owner, repo, alert_number). Verify the advisory and the fixed version are present.
4. Rank the alerts returned by `github_get_dependabot_alerts` by severity and report the fix path per package. Verify no critical or high alert is omitted from the report.
5. Dismiss or reopen only on explicit confirmation with `github_update_dependabot_alert`(owner, repo, alert_number, state). Verify by calling `github_get_dependabot_alert` and checking the new state.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| HTTP 401 or 403 from a github_* tool | `GITHUB_TOKEN` is missing, expired or lacks the scope for this owner | Set a valid `GITHUB_TOKEN` with the needed scope in the server environment, and call `github_get_repository`(owner, repo) to verify it returns the repository |
| 403 listing alerts | Dependabot alerts are disabled or the token lacks security events scope | Enable Dependabot alerts for the repository and use a token with the security_events scope, and call `github_get_dependabot_alerts`(owner, repo) to verify |
| Dismiss rejected | state is dismissed but dismissed_reason is missing | Pass dismissed_reason and dismissed_comment to `github_update_dependabot_alert` and verify the state changed |

## Task workflows

### review-dependabot-alerts

Read and dismiss Dependabot alerts of a repository or organization. Contract steps, in order:

1. `github_get_dependabot_alert`(owner, repo, alert_number)
2. `github_get_dependabot_alerts`(owner, repo; optional params_json)
3. `github_get_org_dependabot_alerts`(org; optional params_json)
4. `github_update_dependabot_alert`(owner, repo, alert_number, state; optional dismissed_reason, dismissed_comment)
