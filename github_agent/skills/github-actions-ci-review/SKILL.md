---
name: github-actions-ci-review
skill_type: skill
description: >-
  Review GitHub Actions CI results across an organization, a user, or a single repository via the github-agent MCP server — enumerate repos, list workflow runs (filterable to failures), drill into a run's jobs, and read a job's logs. Use when the agent must check CI health for an org/user/repo, find red pipelines, or read why a run/job failed. Do NOT use to gate a single PR's merge on CI (github-pull-request-review), to trigger a deploy workflow (github-repository-management), or to manage issues (github-issue-tracking).
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Actions Ci Review

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Task workflows

### review-ci-results

Read workflow runs, jobs and logs of a repository. Contract steps, in order:

1. `github_get_workflow_job_logs`(owner, repo, job_id)
2. `github_get_workflow_run`(owner, repo, run_id)
3. `github_get_workflow_run_jobs`(owner, repo, run_id; optional params_json)
4. `github_get_workflow_runs`(owner, repo; optional max_pages, page, per_page, api_parameters, status, branch)
5. `github_get_workflows`(owner, repo)
6. `github_ingest_pipelines`(owner, repo; optional status, branch, per_page, max_pages, include_jobs, repo_node_id)

### run-ci

Trigger, re-run, cancel and delete workflow runs. Contract steps, in order:

1. `github_cancel_workflow_run`(owner, repo, run_id)
2. `github_delete_workflow_run`(owner, repo, run_id; optional confirm) (destructive: pass confirm=true only after the person agrees)
3. `github_rerun_workflow_run`(owner, repo, run_id)
4. `github_trigger_workflow_dispatch`(owner, repo, workflow_id, ref; optional inputs)
