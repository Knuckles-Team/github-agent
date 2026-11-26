---
name: github-ci-failure-sweep
skill_type: skill
description: >-
  Sweeps GitHub Actions across one or more GitHub accounts (a user and/or an organization, all repositories) and reports which CI pipelines are currently failing, with the failing workflow, last failed run, and a probable cause. Use when asked which CI/pipelines are red, to audit Actions health across repos, to find failing builds for a user/org, or to triage broken CI before fixing it.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Ci Failure Sweep

Scope: Sweeps GitHub Actions across one or more GitHub accounts (a user and/or an organization, all repositories) and reports which CI pipelines are currently failing, with the failing workflow, last failed run, and a probable cause. Use when asked which CI/pipelines are red, to audit Actions health across repos, to find failing builds for a user/org, or to triage broken CI before fixing it. Defaults to read-only analysis; re-running or cancelling runs is opt-in and runs only on explicit confirmation. Do NOT use for inspecting a single PR's checks (use github-tools) or for GitLab pipelines.

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Steps

1. List the repositories in scope with `github_get_org_repos`(org) or `github_get_repositories`(owner). Verify the list is non-empty.
2. For each repository read the latest runs with `github_get_workflow_runs`(owner, repo, status="failure"). Verify only runs with a failing conclusion are kept.
3. Save the runs JSON and summarize it with `python scripts/summarize_runs.py runs.json`. Verify the summary lists each failing workflow once with its last failed run.
4. Find a probable cause using `github_get_workflow_run_jobs`(owner, repo, run_id) and `github_get_workflow_job_logs`(owner, repo, job_id). Verify each cause quotes a log line.
5. Report the failing pipelines. Only on explicit confirmation call `github_rerun_workflow_run`(owner, repo, run_id) and verify the new run appears in `github_get_workflow_runs`(owner, repo).

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| HTTP 401 or 403 from a github_* tool | `GITHUB_TOKEN` is missing, expired or lacks the scope for this owner | Set a valid `GITHUB_TOKEN` with the needed scope in the server environment, and call `github_get_repository`(owner, repo) to verify it returns the repository |
| Summary script prints nothing | The runs JSON was empty or not the list_runs shape | Re-fetch with `github_get_workflow_runs`(owner, repo) and pass that exact JSON to `scripts/summarize_runs.py` |
| Rerun refused | The run is still in progress or the token lacks actions write scope | Wait for completion by polling `github_get_workflow_run`(owner, repo, run_id) and use a token with actions write access |

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
