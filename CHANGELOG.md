# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.0] - 2026-11-12

### Changed
- Destructive operations use the SDK's one declaration and guard: `merge_pull_request` and `update_dependabot_alert` are declared destructive, auto-merge asks for confirmation first, and every destructive tool takes the standard `confirm` argument. The connector-local `GITHUB_ALLOW_DESTRUCTIVE` setting and `allow_destructive_default` are removed; the one operator setting is `SCION_ALLOW_DESTRUCTIVE`.
- Require `scion-sdk[mcp,graphql]>=0.10.2,<0.11`.

## [2.4.15] - 2026-11-06

### Changed
- Connector surface: agent-definition prompts (operator, release manager, repository administrator, triage lead), task skills that name the served tools, tool-use contracts for all 101 tools, data locations and roles.
- Require `scion-sdk[mcp,graphql]>=0.9.0`; `uv.lock` relocked.
- The condensed action-routed `github_<domain>` tool modules and their tests are removed; `github_pulls_enable_auto_merge` and `github_pulls_disable_auto_merge` are typed tools, and the ingestion tools take typed arguments instead of `params_json`.

## [Unreleased]

### Added
-

### Changed
- Require scion-sdk>=0.6.4,<0.7.0; tests lint-clean (bare except, import order).

### Fixed
- `create_repository` (`github_agent/api/api_client_repos.py`) silently
  ignored an `org` keyword argument and always POSTed to `/user/repos`,
  creating the repository under the caller's personal account even when an
  organization was explicitly requested. GitHub's `/user/repos` endpoint
  accepts an unrecognized `org` body field without erroring, so this was a
  silent wrong-target bug (a `201 Created` response, just in the wrong
  place) rather than a clean failure. `create_repository` now pops `org`
  out of kwargs and, when present, POSTs to `/orgs/{org}/repos` instead
  (`org` is used only as the URL path segment and is never sent in the
  JSON body) — matching the existing `create_organization_repository`
  endpoint. Calling with no `org` kwarg is unchanged (still `/user/repos`).

## [0.2.55] - 2026-04-29

### Added
- Initial release
