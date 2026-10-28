---
name: github-agent-development
skill_type: skill
domain: development
description: >-
  Develop, test, gate, and release the github-agent connector. Use this skill when
  changing the connector's API mappings, MCP tools, packaging, documentation,
  or repository gates.
license: MIT
tags: [github-agent, development, connector, test, gates]
tier: repository
metadata:
  author: Genius
  version: '0.2.0'
---

# github-agent development

This repository owns the GitHub connector: the REST client (`github_agent/api/api_client_<domain>.py`), the GraphQL client (`github_gql.py`), HTTP policy (`github_http.py`), auth, and the MCP tool domains under `github_agent/mcp/`. Shared connector contracts and transports belong to `scion-sdk`;
admission, serving, and durable graph state belong to their owning
repositories. Read `AGENTS.md` before changing code and keep provider
credentials and endpoints out of source, logs, and skill content.

## Rapid delivery: the 5-minute contract

Specs are already designed. Deliver small, reviewable slices and cite the
owning spec row in the commit trailer.

- Find the owning row first with `git grep -n "<term>" -- specs` and extend
  that spec before adding new requirement text.
- Reuse existing SDK clients, transport policies, and tool factories. Do not
  create a second implementation of a connector capability.
- Keep changes scoped to this connector. A change owned by the SDK, Arbora,
  or the knowledge graph belongs in that repository instead.
- Every delivered requirement needs a bound test. Generated `status.json`
  files are produced by `pipelines-hook spec-status`; never hand-edit them.
- One logical change per commit, with `Spec: <ID>` and the assisting agent
  trailer adjacent in the final paragraph.

## Setup and gates

Use the pinned environment from `uv sync --locked --all-extras`. Run the
repository gate from the checkout root:

```bash
uvx pre-commit run --all-files
```

For focused feedback, run the relevant tests with the repository's pinned
environment, for example:

```bash
uv run --all-extras pytest tests/test_action_discovery.py tests/test_default_content_surfaces.py -q
uv run --all-extras pytest tests/test_github_http.py tests/test_github_gql.py -q
uv run --all-extras pytest tests/test_kg_ingest.py tests/test_fleet_name.py -q
```

The canonical `.pre-commit-config.yaml` delegates connector-standard,
secrets, spec, tests-fast, and dependency-readiness checks to
`Knuckles-Team/pipelines` at `rev: main`. Do not replace those shared hooks
with local copies or bypass a failing check.

## Connector-specific checks

- Credentials come from `GITHUB_TOKEN`; the endpoint is derived from `GITHUB_URL`
  (`api.github.com`, or a GitHub Enterprise `/api/v3` host whose GraphQL endpoint
  is `/api/graphql`). Mock all GitHub calls; never use a real token in a gate.
- Destructive operations stay behind `GITHUB_ALLOW_DESTRUCTIVE`; keep it false
  by default and test both states.
- Prefer one aliased `github_graphql` query over many REST calls for cross-repo
  reads; keep REST and GraphQL parity when adding a tool domain.
- Packaged skills under `github_agent/skills/` have their own helper tests
  (`tests/test_skill_*.py`); update them with the skill scripts.
- Keep the MCP surface aligned with `scion-sdk`; read the version from package
  metadata rather than spelling it in a module.
- Keep `github_agent/skills/` package data intact when adding skills.
- Documentation lives under `docs/` and is published by the shared Pages
  workflow; source status records live under `specs/`.

## Delivery discipline

Work in a real Git worktree based on `origin/main`. Inspect the complete diff,
stage only reviewed paths, and verify `git status --short` after every gate.
Do not use `git stash`, `git add -A`, or `git add .`; do not claim Pages
publication until a successful exact-source workflow run and the public page
are both observed.
