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

## Steps

1. Check whether the repository exists with `github_get_repository`(owner, repo). Verify the 200 or 404 outcome decides create versus update.
2. Create it with `github_create_repository`(name) or update it with `github_update_repository`(owner, repo). Verify with `github_get_repository` that the settings match.
3. Protect the default branch with `github_update_branch_protection`(owner, repo, branch). Verify with `github_get_branch_protection`(owner, repo, branch).
4. Add the pipeline files with `github_create_content`(owner, repo, path, message, content). Verify with `github_get_contents`(owner, repo, path).
5. Enable Pages with `python scripts/enable_pages.py` or `github_create_pages`(owner, repo). Verify with `github_get_pages`(owner, repo) that the site reports a build.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| HTTP 401 or 403 from a github_* tool | `GITHUB_TOKEN` is missing, expired or lacks the scope for this owner | Set a valid `GITHUB_TOKEN` with the needed scope in the server environment, and call `github_get_repository`(owner, repo) to verify it returns the repository |
| Create fails because the name exists | The repository already exists under that owner | Switch to `github_update_repository`(owner, repo) and verify the settings instead of creating |
| Pages call rejected | Pages is already enabled or no workflow produced a build | Use `github_update_pages`(owner, repo) for an existing site and check `github_list_pages_builds`(owner, repo) |

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
