# github-agent

GitHub **MCP Server + A2A Agent** for the agent-utilities ecosystem — a typed,
deterministic tool surface and supervisor agent over the GitHub REST API.

!!! info "Official documentation"
    This site is the canonical reference for `github-agent`, maintained alongside every
    release.

[![PyPI](https://img.shields.io/pypi/v/github-agent)](https://pypi.org/project/github-agent/)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
[![License](https://img.shields.io/pypi/l/github-agent)](https://github.com/Knuckles-Team/github-agent/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/Knuckles-Team/github-agent)

## Overview

`github-agent` wraps the GitHub REST API with typed, deterministic MCP tools and a
Pydantic-AI supervisor agent. It provides:

- **`Api`** — a `requests`-based REST client (`github_agent.api_client.Api`) composed
  from per-domain mixins (repositories, issues, pull requests, branches, commits,
  contents, organizations, collaborators, releases, workflows, search), with
  parallel pagination and tolerant error handling.
- **Action-dispatch MCP tools** — eleven tool domains (`github_repos`,
  `github_issues`, `github_pulls`, …), each toggled by an environment switch and
  routed by an `action` argument.
- **An A2A supervisor agent** (`github-agent` console script) that delegates GitHub
  work to specialized child agents.

The server remains inactive when credentials are absent — a missing `GITHUB_TOKEN`
leaves calls unauthenticated rather than raising at import time.

## Explore the documentation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](installation.md)** — pip, source, extras, and the prebuilt Docker image.
- :material-server-network: **[Deployment](deployment.md)** — run the MCP and agent servers, Docker Compose, Caddy + Technitium.
- :material-console: **[Usage](usage.md)** — the MCP tools, the `Api` client, and the CLI.
- :material-sitemap: **[Overview](overview.md)** — supervisor architecture and specialized agents.
- :material-tag-multiple: **[Concepts](concepts.md)** — the `CONCEPT:GH-*` registry.

</div>

## Quick start

```bash
pip install github-agent
github-mcp                       # stdio MCP server (default transport)
```

Connect it to GitHub:

```bash
export GITHUB_URL=https://api.github.com
export GITHUB_TOKEN="<GITHUB_TOKEN>"
github-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

See **[Installation](installation.md)** and **[Deployment](deployment.md)** for the
full matrix (PyPI extras, Docker image, all transports, the agent server, reverse
proxy, DNS).
