# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
-

### Changed
-

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
