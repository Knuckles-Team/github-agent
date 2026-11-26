---
name: github-actions-ci-review
skill_type: skill
description: >-
  Review GitHub Actions CI results across an organization, a user, or a single repository via the github-agent MCP server — enumerate repos, list workflow runs (filterable to failures), drill into a run's jobs, and read a job's logs. Use when the agent must check CI health for an org/user/repo, find red pipelines, or read why a run/job failed.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Actions Ci Review

Scope: Review GitHub Actions CI results across an organization, a user, or a single repository via the github-agent MCP server — enumerate repos, list workflow runs (filterable to failures), drill into a run's jobs, and read a job's logs. Use when the agent must check CI health for an org/user/repo, find red pipelines, or read why a run/job failed. Do NOT use to gate a single PR's merge on CI (github-pull-request-review), to trigger a deploy workflow (github-repository-management), or to manage issues (github-issue-tracking).

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Steps

1. Resolve the target with `github_get_repositories`(owner) for an org or user, or take the given owner and repo. Verify each repository name exists in the result.
2. List recent runs with `github_get_workflow_runs`(owner, repo, status="failure"). Verify every returned run carries a run_id and a conclusion.
3. Drill into one failing run with `github_get_workflow_run_jobs`(owner, repo, run_id). Verify the failed job and its job_id are identified.
4. Read the failure with `github_get_workflow_job_logs`(owner, repo, job_id). Verify the log shows the failing step and its error text.
5. Report each red workflow with its run_id, failed job and cause. Only on request call `github_rerun_workflow_run`(owner, repo, run_id) and verify the new run appears in `github_get_workflow_runs`(owner, repo).

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| HTTP 401 or 403 from a github_* tool | `GITHUB_TOKEN` is missing, expired or lacks the scope for this owner | Set a valid `GITHUB_TOKEN` with the needed scope in the server environment, and call `github_get_repository`(owner, repo) to verify it returns the repository |
| Empty run list | The status or branch filter excludes everything, or the repository has no workflows | Call `github_get_workflows`(owner, repo) to confirm workflows exist and drop the status or branch filter |
| Job logs return 404 or empty | The job_id belongs to another run or the logs expired | Re-read the job_id from `github_get_workflow_run_jobs`(owner, repo, run_id) and call `github_get_workflow_job_logs` with that id |

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
