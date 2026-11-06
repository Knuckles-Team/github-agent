---
name: github-ci-failure-sweep
skill_type: skill
description: >-
  Sweeps GitHub Actions across one or more GitHub accounts (a user and/or an organization, all repositories) and reports which CI pipelines are currently failing, with the failing workflow, last failed run, and a probable cause. Use when asked which CI/pipelines are red, to audit Actions health across repos, to find failing builds for a user/org, or to triage broken CI before fixing it. Defaults to read-only analysis; re-running or cancelling runs is opt-in and runs only on explicit confirmation. Do NOT use for inspecting a single PR's checks (use github-tools) or for GitLab pipelines.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Ci Failure Sweep

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
