---
name: github-agent-operations
skill_type: skill
description: >-
  Operate github-agent through its governed MCP and Arbora capabilities, including github actions ci review, github backlog planner, github ci failure sweep, github dependabot review, github issue tracking, github org remediation loop, github pr review sweep, github project provisioner, and related workflows. Use when a request requires this provider's read, change, automation, ingestion, troubleshooting, or evidence workflows.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Agent Operations

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

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
