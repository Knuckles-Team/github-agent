---
name: github-tools
skill_type: skill
description: >-
  Consolidated skill for GitHub workflows and git practices. Use when managing PR comments, fixing CI failures, using the gh CLI for issues/PRs/workflow runs, finishing development branches, performing code review, or managing git worktrees. Do NOT use for general git operations that don't involve GitHub (use standard git commands instead).
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Tools

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Task workflows

### search-code

Search code across repositories. Contract steps, in order:

1. `github_search_code`(q; optional max_pages, page, per_page, api_parameters, sort, order)

### query-graphql

Run a GraphQL query or discover the schema for fleet-wide reads. Contract steps, in order:

1. `github_discover_graphql_schema`(; optional type_name)
2. `github_graphql`(query; optional variables, operation_name)

### publish-a-release

Create, change and delete releases. Contract steps, in order:

1. `github_create_release`(owner, repo, tag_name; optional name, body, draft, prerelease)
2. `github_delete_release`(owner, repo, release_id; optional confirm) (destructive: pass confirm=true only after the person agrees)
3. `github_get_release`(owner, repo, release_id)
4. `github_get_releases`(owner, repo)
5. `github_update_release`(owner, repo, release_id; optional params_json)

### ingest-repositories

Mirror repositories into the knowledge graph. Contract steps, in order:

1. `github_get_repositories`(; optional max_pages, page, per_page, api_parameters, owner, repo)
2. `github_ingest_repos`(; optional visibility, affiliation, type)
