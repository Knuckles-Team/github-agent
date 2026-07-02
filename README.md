# Github Agent
## CLI or API | MCP | Agent

![PyPI - Version](https://img.shields.io/pypi/v/github-agent)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/github-agent)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/github-agent)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/github-agent)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/github-agent)
![PyPI - License](https://img.shields.io/pypi/l/github-agent)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/github-agent)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/github-agent)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/github-agent)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/github-agent)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/github-agent)
![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/github-agent)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/github-agent)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/github-agent)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/github-agent)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/github-agent)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/github-agent)

*Version: 1.0.1*

> **Documentation** — Installation, deployment, and usage across the MCP, API, and
> CLI interfaces, including the integrated A2A agent server, are maintained in the
> [official documentation](https://knuckles-team.github.io/github-agent/).

---

## Overview

**Github Agent** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with GitHub Agent for MCP.

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Enterprise-Grade Security:** Comprehensive support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Integrated Graph Agent:** Built-in Pydantic AI agent supporting the Agent Control Protocol (ACP) and standard Web interfaces (AG-UI).
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.

---

## CLI or API

This agent wraps the GitHub Agent for MCP API. You can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

_Auto-generated from the live MCP server — do not edit by hand._

<!-- MCP-TOOLS-TABLE:START -->

#### Condensed action-routed tools (default — `MCP_TOOL_MODE=condensed`)

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `github_actions` | `ACTIONTOOL` | Manage GitHub actions workflows and runs. |
| `github_branches` | `BRANCHTOOL` | Manage GitHub branches. |
| `github_collaborators` | `COLLABORATORTOOL` | Manage repository collaborators. |
| `github_commits` | `COMMITTOOL` | Manage GitHub commits. |
| `github_contents` | `CONTENTTOOL` | Manage GitHub contents. |
| `github_issues` | `ISSUETOOL` | Manage GitHub issues. |
| `github_orgs` | `ORGTOOL` | Manage GitHub organizations. |
| `github_pulls` | `PULLTOOL` | Manage GitHub pull requests. |
| `github_releases` | `RELEASETOOL` | Manage repository releases. |
| `github_repos` | `REPOTOOL` | Manage GitHub repositories and their GitHub Pages sites. |
| `github_search` | `SEARCHTOOL` | Search GitHub repositories, issues, or code. |

#### Verbose 1:1 API-mapped tools (`MCP_TOOL_MODE=verbose` or `both`)

<details>
<summary>66 per-operation tools — one per public API method (click to expand)</summary>

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `github_add_collaborator` | `APITOOL` | Add a collaborator to a repository. |
| `github_cancel_workflow_run` | `APITOOL` | Cancel a workflow run. |
| `github_create_branch` | `APITOOL` | Create a new branch in a repository (using git ref creation). |
| `github_create_content` | `APITOOL` | Create a file in a repository. |
| `github_create_issue` | `APITOOL` | Create a new issue in a repository. |
| `github_create_or_update_repo_secret` | `APITOOL` | Create or update a repository Actions secret. |
| `github_create_organization` | `APITOOL` | Create an organization — GitHub Enterprise Server ONLY. |
| `github_create_organization_repository` | `APITOOL` | Create a repository in an organization. |
| `github_create_pages` | `APITOOL` | Enable GitHub Pages for a repository (HTTP 201). |
| `github_create_pull_request` | `APITOOL` | Create a new pull request in a repository. |
| `github_create_release` | `APITOOL` | Create a new repository release. |
| `github_create_repository` | `APITOOL` | Create a new repository for the authenticated user. |
| `github_delete_branch` | `APITOOL` | Delete a branch in a repository. |
| `github_delete_branch_protection` | `APITOOL` | Delete branch protection configuration. |
| `github_delete_content` | `APITOOL` | Delete a file in a repository. |
| `github_delete_organization` | `APITOOL` | Schedule an organization for deletion. IRREVERSIBLE. |
| `github_delete_pages` | `APITOOL` | Disable GitHub Pages for a repository and delete the site. |
| `github_delete_release` | `APITOOL` | Delete a repository release. |
| `github_delete_repo_secret` | `APITOOL` | Delete a repository Actions secret. |
| `github_delete_repository` | `APITOOL` | Delete a repository. |
| `github_delete_workflow_run` | `APITOOL` | Delete a workflow run. |
| `github_get_branch` | `APITOOL` | Get a single branch in a repository. |
| `github_get_branch_protection` | `APITOOL` | Get branch protection configuration. |
| `github_get_branches` | `APITOOL` | List branches for a repository. |
| `github_get_collaborators` | `APITOOL` | List collaborators for a repository. |
| `github_get_commit` | `APITOOL` | Get a single commit in a repository. |
| `github_get_commits` | `APITOOL` | List commits for a repository. |
| `github_get_contents` | `APITOOL` | Get contents of a file or directory in a repository. |
| `github_get_issue` | `APITOOL` | Get a single issue in a repository. |
| `github_get_issues` | `APITOOL` | List issues for a repository. |
| `github_get_org_members` | `APITOOL` | List members for an organization. |
| `github_get_org_repos` | `APITOOL` | List repositories for an organization. |
| `github_get_org_teams` | `APITOOL` | List teams for an organization. |
| `github_get_organization` | `APITOOL` | Get an organization's full profile. |
| `github_get_organization_membership` | `APITOOL` | Get a user's organization membership (state and role). |
| `github_get_pages` | `APITOOL` | Get the GitHub Pages site configuration for a repository. |
| `github_get_pages_build_latest` | `APITOOL` | Get the latest GitHub Pages build for a repository. |
| `github_get_pull_request` | `APITOOL` | Get a single pull request. |
| `github_get_pull_requests` | `APITOOL` | List pull requests for a repository. |
| `github_get_release` | `APITOOL` | Get a single repository release. |
| `github_get_releases` | `APITOOL` | List repository releases. |
| `github_get_repo_secrets` | `APITOOL` | List repository Actions secrets names. |
| `github_get_repositories` | `APITOOL` | List repositories for the authenticated user. |
| `github_get_repository` | `APITOOL` | Get a specific repository. |
| `github_get_workflow_run` | `APITOOL` | Get a single workflow run. |
| `github_get_workflow_runs` | `APITOOL` | List workflow runs for a repository. |
| `github_get_workflows` | `APITOOL` | List workflows for a repository. |
| `github_list_organizations` | `APITOOL` | List organizations. |
| `github_list_pages_builds` | `APITOOL` | List GitHub Pages builds for a repository (newest first). |
| `github_remove_collaborator` | `APITOOL` | Remove a collaborator from a repository. |
| `github_remove_organization_member` | `APITOOL` | Remove a user from an organization (repositories access included). |
| `github_request_pages_build` | `APITOOL` | Request a fresh GitHub Pages build without pushing a commit. |
| `github_rerun_workflow_run` | `APITOOL` | Re-run a workflow run. |
| `github_search_code` | `APITOOL` | Search code using query keywords. |
| `github_search_issues` | `APITOOL` | Search issues using query keywords. |
| `github_search_repositories` | `APITOOL` | Search repositories using query keywords. |
| `github_set_organization_membership` | `APITOOL` | Add a user to an organization or update their role. |
| `github_trigger_workflow_dispatch` | `APITOOL` | Trigger a workflow dispatch event. |
| `github_update_branch_protection` | `APITOOL` | Update branch protection configuration. |
| `github_update_content` | `APITOOL` | Update a file in a repository. |
| `github_update_issue` | `APITOOL` | Update an issue in a repository. |
| `github_update_organization` | `APITOOL` | Update an organization's profile and member settings. |
| `github_update_pages` | `APITOOL` | Update the GitHub Pages configuration for a repository. |
| `github_update_pull_request` | `APITOOL` | Update a pull request. |
| `github_update_release` | `APITOOL` | Update (PATCH) a repository release. |
| `github_update_repository` | `APITOOL` | Update a repository. |

</details>

_11 action-routed tool(s) (default) · 66 verbose 1:1 tool(s). Each is enabled unless its `<DOMAIN>TOOL` toggle is set false; `MCP_TOOL_MODE` selects the surface (`condensed` default · `verbose` 1:1 · `both`). Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](docs/mcp.md).

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows you to restrict the set of exposed tools in order to prevent blowing up the LLM's context window.

You can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to your transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are supplied, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

<!-- MCP-CONFIG-EXAMPLES:START -->

> **Install the slim `[mcp]` extra.** All examples install `github-agent[mcp]` — the
> MCP-server extra that pulls only the FastMCP / FastAPI tooling (`agent-utilities[mcp]`).
> It deliberately **excludes** the heavy agent runtime (`pydantic-ai`, the epistemic-graph
> engine, `dspy`, `llama-index`), so `uvx` / container installs are far smaller. Use the
> full `[agent]` extra only when you need the integrated Pydantic AI agent.

#### stdio Transport (local IDEs — Cursor, Claude Desktop, VS Code)

```json
{
  "mcpServers": {
    "github-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "github-agent[mcp]",
        "github-mcp"
      ],
      "env": {
        "MCP_TOOL_MODE": "condensed",
        "ACTIONTOOL": "True",
        "BRANCHTOOL": "True",
        "COLLABORATORTOOL": "True",
        "COMMITTOOL": "True",
        "CONTENTTOOL": "True",
        "GITHUB_ALLOW_DESTRUCTIVE": "False",
        "GITHUB_HTTP_CONNECT_TIMEOUT": "10",
        "GITHUB_HTTP_READ_TIMEOUT": "30",
        "GITHUB_TOKEN": "your_github_token_here",
        "GITHUB_URL": "https://api.github.com",
        "GITHUB_VERIFY": "True",
        "ISSUETOOL": "True",
        "ORGTOOL": "True",
        "PULLTOOL": "True",
        "RELEASETOOL": "True",
        "REPOTOOL": "True",
        "SEARCHTOOL": "True"
      }
    }
  }
}
```

#### Streamable-HTTP Transport (networked / production)

```json
{
  "mcpServers": {
    "github-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "github-agent[mcp]",
        "github-mcp",
        "--transport",
        "streamable-http",
        "--port",
        "8000"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "MCP_TOOL_MODE": "condensed",
        "ACTIONTOOL": "True",
        "BRANCHTOOL": "True",
        "COLLABORATORTOOL": "True",
        "COMMITTOOL": "True",
        "CONTENTTOOL": "True",
        "GITHUB_ALLOW_DESTRUCTIVE": "False",
        "GITHUB_HTTP_CONNECT_TIMEOUT": "10",
        "GITHUB_HTTP_READ_TIMEOUT": "30",
        "GITHUB_TOKEN": "your_github_token_here",
        "GITHUB_URL": "https://api.github.com",
        "GITHUB_VERIFY": "True",
        "ISSUETOOL": "True",
        "ORGTOOL": "True",
        "PULLTOOL": "True",
        "RELEASETOOL": "True",
        "REPOTOOL": "True",
        "SEARCHTOOL": "True"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed Streamable-HTTP instance by `url`:

```json
{
  "mcpServers": {
    "github-mcp": {
      "url": "http://localhost:8000/github-mcp/mcp"
    }
  }
}
```

Deploying the Streamable-HTTP server via Docker:

```bash
docker run -d \
  --name github-mcp-mcp \
  -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e MCP_TOOL_MODE=condensed \
  -e ACTIONTOOL=True \
  -e BRANCHTOOL=True \
  -e COLLABORATORTOOL=True \
  -e COMMITTOOL=True \
  -e CONTENTTOOL=True \
  -e GITHUB_ALLOW_DESTRUCTIVE=False \
  -e GITHUB_HTTP_CONNECT_TIMEOUT=10 \
  -e GITHUB_HTTP_READ_TIMEOUT=30 \
  -e GITHUB_TOKEN=your_github_token_here \
  -e GITHUB_URL=https://api.github.com \
  -e GITHUB_VERIFY=True \
  -e ISSUETOOL=True \
  -e ORGTOOL=True \
  -e PULLTOOL=True \
  -e RELEASETOOL=True \
  -e REPOTOOL=True \
  -e SEARCHTOOL=True \
  knucklessg1/github-agent:mcp
```

_Auto-generated from the code-read env surface (`MCP_TOOL_MODE` + package vars) — do not edit._
<!-- MCP-CONFIG-EXAMPLES:END -->

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`github-agent` can also run as a **local container** (Docker / Podman / `uv`) or be
consumed from a **remote deployment**. The
[Deployment guide](https://knuckles-team.github.io/github-agent/deployment/) has full, copy-paste
`mcp_config.json` for all four transports — **stdio**, **streamable-http**,
**local container / uv**, and **remote URL**:

- **Local container / uv** — launch the server from `mcp_config.json` via `uvx`,
  `docker run`, or `podman run`, or point at a local streamable-http container by `url`.
- **Remote URL** — connect to a server deployed behind Caddy at
  `http://github-mcp.arpa/mcp` using the `"url"` key.
<!-- END GENERATED: additional-deployment-options -->

---

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `TRANSPORT` | `stdio` | options: stdio, streamable-http, sse |
| `ENABLE_OTEL` | `True` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:8080/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` | `pk-...` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY` | `sk-...` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `EUNOMIA_TYPE` | `none` | options: none, embedded, remote |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` |  |
| `EUNOMIA_REMOTE_URL` | `http://eunomia-server:8000` |  |
| `GITHUB_URL` | `https://api.github.com` |  |
| `GITHUB_VERIFY` | `True` |  |
| `GITHUB_SSL_VERIFY` | `True` | GITHUB_SSL_VERIFY takes precedence over GITHUB_VERIFY when set |
| `GITHUB_ALLOW_DESTRUCTIVE` | `False` | Allow destructive (delete/force) operations |
| `GITHUB_HTTP_CONNECT_TIMEOUT` | `10` | HTTP client timeouts (seconds) |
| `GITHUB_HTTP_READ_TIMEOUT` | `30` |  |
| `DEBUG` | `False` |  |
| `PYTHONUNBUFFERED` | `1` |  |
| `GITHUB_TOKEN` | `your_github_token_here` |  |
| `ACTIONTOOL` | `True` | MCP tools table (condensed action-routed surface). |
| `BRANCHTOOL` | `True` |  |
| `COLLABORATORTOOL` | `True` |  |
| `COMMITTOOL` | `True` |  |
| `CONTENTTOOL` | `True` |  |
| `ISSUETOOL` | `True` |  |
| `ORGTOOL` | `True` |  |
| `PULLTOOL` | `True` |  |
| `RELEASETOOL` | `True` |  |
| `REPOTOOL` | `True` |  |
| `SEARCHTOOL` | `True` |  |

#### Inherited agent-utilities variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `MCP_TOOL_MODE` | `condensed` | Tool surface: `condensed` | `verbose` | `both` |
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `MCP_CLIENT_AUTH` | — | Outbound MCP auth (`oidc-client-credentials` for fleet calls) |
| `OIDC_CLIENT_ID` | — | OIDC client id (service-account auth) |
| `OIDC_CLIENT_SECRET` | — | OIDC client secret (service-account auth) |
| `MCP_URL` | `http://localhost:8000/mcp` | URL of the MCP server the agent connects to |
| `PROVIDER` | `openai` | LLM provider for the agent |
| `MODEL_ID` | `gpt-4o` | Model id for the agent |
| `ENABLE_WEB_UI` | `True` | Serve the AG-UI web interface |

_31 package + 12 inherited variable(s). Auto-generated from `.env.example` + the shared agent-utilities set — do not edit._
<!-- ENV-VARS-TABLE:END -->


Every variable the server reads.

### Connection & Credentials
| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_URL` | GitHub REST API base URL (use your GitHub Enterprise Server URL for self-hosted) | `https://api.github.com` |
| `GITHUB_TOKEN` | GitHub Personal Access Token / OAuth token (`Authorization: Bearer <token>`) | — |
| `GITHUB_VERIFY` | TLS certificate verification | `True` |

### MCP server / transport
| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSPORT` | `stdio`, `streamable-http`, or `sse` | `stdio` |
| `HOST` | Bind host (HTTP transports) | `0.0.0.0` |
| `PORT` | Bind port (HTTP transports) | `8000` |
| `MCP_TOOL_MODE` | Tool surface: `condensed`, `verbose`, or `both` | `condensed` |
| `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS` | Comma-separated tool allow/deny list | — |
| `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS` | Comma-separated tag allow/deny list | — |
| `DEBUG` | Verbose logging | `False` |
| `PYTHONUNBUFFERED` | Unbuffered stdout (recommended in containers) | `1` |

### Tool toggles
Each action-routed tool can be disabled individually via its toggle env var (set to `false`).
The full list is in the [Available MCP Tools](#available-mcp-tools) table above.

| Variable | Tool | Default |
|----------|------|---------|
| `ACTIONTOOL` | `github_actions` — Actions workflows and runs | `True` |
| `BRANCHTOOL` | `github_branches` — branches | `True` |
| `COLLABORATORTOOL` | `github_collaborators` — repository collaborators | `True` |
| `COMMITTOOL` | `github_commits` — commits | `True` |
| `CONTENTTOOL` | `github_contents` — file contents | `True` |
| `ISSUETOOL` | `github_issues` — issues | `True` |
| `ORGTOOL` | `github_orgs` — organizations | `True` |
| `PULLTOOL` | `github_pulls` — pull requests | `True` |
| `RELEASETOOL` | `github_releases` — releases | `True` |
| `REPOTOOL` | `github_repos` — repositories and Pages sites | `True` |
| `SEARCHTOOL` | `github_search` — search repositories, issues, code | `True` |

### Telemetry & governance
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OTEL` | Enable OpenTelemetry export | `True` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | — |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_SECRET_KEY` | OTLP auth keys | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol (e.g. `http/protobuf`) | — |
| `EUNOMIA_TYPE` | Authorization mode: `none`, `embedded`, `remote` | `none` |
| `EUNOMIA_POLICY_FILE` | Embedded policy file | `mcp_policies.json` |
| `EUNOMIA_REMOTE_URL` | Remote Eunomia server URL | — |

### Agent CLI (full `[agent]` runtime only)
| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_URL` | URL of the MCP server the agent connects to | `http://localhost:8000/mcp` |
| `PROVIDER` | LLM provider (e.g. `openai`) | `openai` |
| `MODEL_ID` | Model id (e.g. `gpt-4o`) | `gpt-4o` |
| `ENABLE_WEB_UI` | Serve the AG-UI web interface | `True` |

See [`.env.example`](.env.example) for a copy-paste starting point.

## Agent

This repository features a fully integrated Pydantic AI Graph Agent. It communicates over the **Agent Control Protocol (ACP)** and interacts seamlessly with the **Agent Web UI (AG-UI)** and Terminal interface.

### Running the Agent CLI
To start the interactive command-line agent:

```bash
# Set credentials
export GITHUB_URL="your_value"
export GITHUB_VERIFY="your_value"
export DEBUG="your_value"
export PYTHONUNBUFFERED="your_value"
export GITHUB_TOKEN="your_value"

# Run the agent server
github-agent --provider openai --model-id gpt-4o
```

### Docker Compose Orchestration
The following `docker/agent.compose.yml` configures the Agent, Web UI, and Terminal Interface together:

```yaml
version: '3.8'

services:
  github-agent-mcp:
    image: knucklessg1/github-agent:mcp
    container_name: github-agent-mcp
    hostname: github-agent-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  github-agent-agent:
    image: knucklessg1/github-agent:latest
    container_name: github-agent-agent
    hostname: github-agent-agent
    restart: always
    depends_on:
      - github-agent-mcp
    env_file:
      - ../.env
    command: [ "github-agent" ]
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=9016
      - MCP_URL=http://github-agent-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
      - ENABLE_OTEL=True
    ports:
      - "9016:9016"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9016/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

```

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/agent.md](docs/agent.md).

---

## Security & Governance

Built directly upon the enterprise-ready [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) core, standard security parameters are fully supported:

### Access Control & Policy Enforcement
- **Eunomia Policies:** Fine-grained, policy-driven tool authorization. Supports `none`, local `embedded` (`mcp_policies.json`), or centralized `remote` modes.
- **OIDC Token Delegation:** Compliant with RFC 8693 token exchange for flowing authenticating user credentials from Web UI / ACP → Agent → MCP.
- **Scoped Credentials:** Execution context runs restricted to the specific caller identity.

### Runtime Security Grid
| Feature | Functionality | Enablement |
|---------|---------------|------------|
| **Tool Guard** | Sensitivity inspection with human-in-the-loop validation | Enabled by default |
| **Prompt Injection Defense** | Input scanning, repetition monitoring, and recursive loop blocks | Enabled by default |
| **Context Safety Guard** | Stuck-loop detectors and contextual overflow preemptive alerts | Enabled by default |

---

## Installation

Pick the extra that matches what you want to run:

| Extra | Installs | Use when |
|-------|----------|----------|
| `github-agent[mcp]` | Slim MCP server only (`agent-utilities[mcp]` — FastMCP/FastAPI) | You only run the **MCP server** (smallest install / image) |
| `github-agent[agent]` | Full agent runtime (`agent-utilities[agent,logfire]` — Pydantic AI + the epistemic-graph engine) | You run the **integrated agent** |
| `github-agent[all]` | Everything (`mcp` + `agent` + `logfire`) | Development / both surfaces |

```bash
# MCP server only (recommended for tool hosting — slim deps)
uv pip install "github-agent[mcp]"

# Full agent runtime (Pydantic AI + epistemic-graph engine)
uv pip install "github-agent[agent]"

# Everything (development)
uv pip install "github-agent[all]"      # or: python -m pip install "github-agent[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `knucklessg1/github-agent:mcp` | `--target mcp` | `github-agent[mcp]` — **slim**, no engine/`pydantic-ai`/`dspy`/`llama-index`/`tree-sitter` | `github-mcp` |
| `knucklessg1/github-agent:latest` | `--target agent` (default) | `github-agent[agent]` — **full** agent runtime + epistemic-graph engine | `github-agent` |

```bash
docker build --target mcp   -t knucklessg1/github-agent:mcp    docker/   # slim MCP server
docker build --target agent -t knucklessg1/github-agent:latest docker/   # full agent
```

`docker/mcp.compose.yml` runs the slim `:mcp` server; `docker/agent.compose.yml` runs the
agent (`:latest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

The **full agent** (`[agent]` / `:latest`) embeds the **epistemic-graph** engine (pulled in
transitively via `agent-utilities[agent]`). For production — or to share one knowledge graph
across multiple agents — run **epistemic-graph as its own database container** and point the
agent at it instead of embedding it. Deployment recipes (single-node + Raft HA), connection
config, and the full database architecture (with diagrams) are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).
The slim `[mcp]` server does **not** require the database.

---

## Documentation

The complete documentation is published as the
[official documentation site](https://knuckles-team.github.io/github-agent/) and is the
recommended reference for installation, deployment, and day-to-day operation.

| Page | Contents |
|---|---|
| [Installation](https://knuckles-team.github.io/github-agent/installation/) | pip, source, extras, prebuilt Docker image |
| [Deployment](https://knuckles-team.github.io/github-agent/deployment/) | run the MCP and agent servers, Compose, Caddy + Technitium, env config |
| [Usage](https://knuckles-team.github.io/github-agent/usage/) | the MCP tools, the `Api` client, the CLI |
| [Overview](https://knuckles-team.github.io/github-agent/overview/) | supervisor architecture and specialized agents |
| [Concepts](https://knuckles-team.github.io/github-agent/concepts/) | concept registry (`CONCEPT:GH-*`) |

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

---

## Contribute

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`


<!-- BEGIN agent-os-genesis-deploy (generated; do not edit between markers) -->

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*): it picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `github-agent` with agent-os-genesis"**.

| Install mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx github-mcp` · or `uv tool install github-agent` |
| Bare-metal, dev (editable) | `uv pip install -e ".[all]"` · or `pip install -e ".[all]"` |
| Container, prod | deploy `knucklessg1/github-agent:latest` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, dev (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`; edits live on restart) |

Secrets are read-existing + seeded via `vault_sync` — you are only prompted for what's missing.

<!-- END agent-os-genesis-deploy -->
