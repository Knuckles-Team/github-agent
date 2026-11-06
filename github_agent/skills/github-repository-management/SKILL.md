---
name: github-repository-management
skill_type: skill
description: >-
  Repository lifecycle and structure operations on GitHub via the github-agent MCP server — list/read/create/update/delete repositories, manage branches and branch protection, read and write file contents, manage collaborators, and enable/configure GitHub Pages. Use when the agent must inspect or provision a repo, set up branch protection, commit a file, add a collaborator, or turn on Pages. Do NOT use for pull-request review (github-pull-request-review), issue triage (github-issue-tracking), or raw GraphQL fan-out (use the github_graphql tool).
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Repository Management

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

### provision-a-repository

Create a repository, set branch protection, secrets and Pages. Contract steps, in order:

1. `github_create_or_update_repo_secret`(owner, repo, secret_name, encrypted_value, key_id)
2. `github_create_organization_repository`(org, name; optional params_json)
3. `github_create_pages`(owner, repo; optional build_type, source)
4. `github_create_repository`(name; optional params_json)
5. `github_get_branch_protection`(owner, repo, branch)
6. `github_get_pages`(owner, repo)
7. `github_get_pages_build_latest`(owner, repo)
8. `github_get_repo_secret_public_key`(owner, repo)
9. `github_get_repo_secrets`(owner, repo)
10. `github_list_pages_builds`(owner, repo; optional max_pages, page, per_page, api_parameters)
11. `github_request_pages_build`(owner, repo)
12. `github_update_branch_protection`(owner, repo, branch, protection_config)
13. `github_update_pages`(owner, repo; optional params_json)
14. `github_update_repository`(owner, repo; optional params_json)

### retire-a-repository

Delete a repository, secret, branch protection or Pages site. Contract steps, in order:

1. `github_delete_branch_protection`(owner, repo, branch; optional confirm) (destructive: pass confirm=true only after the person agrees)
2. `github_delete_pages`(owner, repo; optional confirm) (destructive: pass confirm=true only after the person agrees)
3. `github_delete_repo_secret`(owner, repo, secret_name; optional confirm) (destructive: pass confirm=true only after the person agrees)
4. `github_delete_repository`(owner, repo; optional confirm) (destructive: pass confirm=true only after the person agrees)

### manage-collaborators

Add and remove repository collaborators. Contract steps, in order:

1. `github_add_collaborator`(owner, repo, username; optional permission)
2. `github_get_collaborators`(owner, repo; optional max_pages, page, per_page, api_parameters, affiliation)
3. `github_remove_collaborator`(owner, repo, username; optional confirm) (destructive: pass confirm=true only after the person agrees)
