---
name: github-project-provisioner
skill_type: skill
description: >-
  Provisions and configures a new or existing GitHub project using standard default settings. Use this skill when asked to create a new project, provision a repository, or set up GitHub Actions pipelines and GitHub Pages.
license: MIT
tags: []
metadata:
  author: Knuckles-Team
  version: '1.0.0'
---
# Github Project Provisioner

Typed github_* tools take owner and repo explicitly; resolve both before acting.

## Rules

- Resolve owner and repo (and the pull request number) before any call.
- Read first, then change, then read the object back.
- Delete, remove and merge-class tools take confirm=true or an explicit request; ask first.

## Task workflows

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
