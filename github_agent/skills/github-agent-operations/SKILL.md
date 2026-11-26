---
name: github-agent-operations
skill_type: skill
description: >-
  Operate github-agent through its governed MCP and Arbora capabilities: find repositories, review CI, publish releases, run GraphQL reads. Use when a request needs this provider's read, change, ingestion or troubleshooting workflows. Do NOT use where a focused github-* skill matches.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Agent Operations

Scope: Operate github-agent through its governed MCP and Arbora capabilities, including github actions ci review, github backlog planner, github ci failure sweep, github dependabot review, github issue tracking, github org remediation loop, github pr review sweep, github project provisioner, and related workflows. Use when a request requires this provider's read, change, automation, ingestion, troubleshooting, or evidence workflows.

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Steps

1. Pick the workflow that matches the request from the Task workflows below (find-repositories, review-ci-results, publish-a-release, query-graphql). Verify with `github_get_repository`(owner, repo) that owner and repo are known before any call.
2. Read first with `github_get_repository`(owner, repo) or `github_get_organization`(org). Verify the response names the expected object.
3. Run the read tools of the chosen workflow, for example `github_get_workflow_runs`(owner, repo) or `github_get_releases`(owner, repo). Verify the result is non-empty and matches the scope asked for.
4. Make a change only when asked, using one write tool such as `github_create_release`(owner, repo, tag_name). Verify by reading the object back with `github_get_release`(owner, repo, release_id).
5. For cross-repository reads use one aliased `github_graphql`(query) call after `github_discover_graphql_schema`(type_name). Verify the response has no errors field.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| HTTP 401 or 403 from a github_* tool | `GITHUB_TOKEN` is missing, expired or lacks the scope for this owner | Set a valid `GITHUB_TOKEN` with the needed scope in the server environment, and call `github_get_repository`(owner, repo) to verify it returns the repository |
| HTTP 404 for a repository, run or number | owner, repo or the numeric id is wrong, or the token cannot see a private repository | Re-resolve the names with `github_search_repositories`(q) and use the exact owner/repo and id it returns |
| Destructive tool refuses to run | confirm=true was not passed or the person has not agreed | Ask the person for explicit agreement, pass confirm=true on that one call and verify the object is gone with the matching read tool |

## Task workflows

### find-repositories

Find repositories and read their details. Contract steps, in order:

1. `github_get_org_repos`(org; optional max_pages, page, per_page, api_parameters, type)
2. `github_get_organization`(org)
3. `github_get_repositories`(; optional max_pages, page, per_page, api_parameters, owner, repo)
4. `github_get_repository`(owner, repo)
5. `github_list_organizations`(; optional max_pages, page, per_page, api_parameters, scope, since)
6. `github_search_repositories`(q; optional max_pages, page, per_page, api_parameters, sort, order)

### review-ci-results

Read workflow runs, jobs and logs of a repository. Contract steps, in order:

1. `github_get_workflow_job_logs`(owner, repo, job_id)
2. `github_get_workflow_run`(owner, repo, run_id)
3. `github_get_workflow_run_jobs`(owner, repo, run_id; optional params_json)
4. `github_get_workflow_runs`(owner, repo; optional max_pages, page, per_page, api_parameters, status, branch)
5. `github_get_workflows`(owner, repo)
6. `github_ingest_pipelines`(owner, repo; optional status, branch, per_page, max_pages, include_jobs, repo_node_id)

### publish-a-release

Create, change and delete releases. Contract steps, in order:

1. `github_create_release`(owner, repo, tag_name; optional name, body, draft, prerelease)
2. `github_delete_release`(owner, repo, release_id; optional confirm) (destructive: pass confirm=true only after the person agrees)
3. `github_get_release`(owner, repo, release_id)
4. `github_get_releases`(owner, repo)
5. `github_update_release`(owner, repo, release_id; optional params_json)

### query-graphql

Run a GraphQL query or discover the schema for fleet-wide reads. Contract steps, in order:

1. `github_discover_graphql_schema`(; optional type_name)
2. `github_graphql`(query; optional variables, operation_name)
