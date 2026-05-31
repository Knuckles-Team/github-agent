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

*Version: 0.23.2*

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
| Tool Module | Toggle Env Var | Enabled by Default | Description & Nested Methods |
|-------------|----------------|--------------------|------------------------------|
| **Repo** | `REPO_TOOL` | `True` | Manage GitHub repositories. Action-routed methods: `create`, `delete`, `get`, `list`, `update`. |
| **Issue** | `ISSUE_TOOL` | `True` | Manage GitHub issues. Action-routed methods: `create`, `get`, `list`, `update`. |
| **Pull** | `PULL_TOOL` | `True` | Manage GitHub pull requests. Action-routed methods: `create`, `get`, `list`, `update`. |
| **Content** | `CONTENT_TOOL` | `True` | Manage GitHub contents. Action-routed methods: `create`, `delete`, `get`, `update`. |
| **Branch** | `BRANCH_TOOL` | `True` | Manage GitHub branches. Action-routed methods: `create`, `delete`, `get`, `list`. |
| **Commit** | `COMMIT_TOOL` | `True` | Manage GitHub commits. Action-routed methods: `get`, `list`. |
| **Search** | `SEARCH_TOOL` | `True` | Search GitHub repositories, issues, or code. Action-routed methods: `code`, `issues`, `repositories`. |
| **Org** | `ORG_TOOL` | `True` | Manage GitHub organizations. Action-routed methods: `members`, `repos`, `teams`. |
| **Collaborator** | `COLLABORATOR_TOOL` | `True` | Manage repository collaborators. Action-routed methods: `add`, `list`, `remove`. |
| **Action** | `ACTION_TOOL` | `True` | Manage GitHub actions workflows and runs. Action-routed methods: `cancel`, `delete_run`, `get_run`, `list_runs`, `list_workflows`, `rerun`, `trigger_dispatch`. |
| **Release** | `RELEASE_TOOL` | `True` | Manage repository releases. Action-routed methods: `create`, `delete`, `get`, `list`, `update`. |

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

#### stdio Transport (Recommended for local IDEs e.g., Cursor, Claude Desktop)
Configure your IDE's `mcp.json` to launch the MCP server via `uvx`:

```json
{
  "mcpServers": {
    "github-agent": {
      "command": "uvx",
      "args": [
        "--from",
        "github-agent",
        "github-mcp"
      ],
      "env": {
        "GITHUB_URL": "your_github_url_here",
        "GITHUB_VERIFY": "your_github_verify_here",
        "DEBUG": "your_debug_here",
        "PYTHONUNBUFFERED": "your_pythonunbuffered_here",
        "GITHUB_TOKEN": "your_github_token_here"
      }
    }
  }
}
```

#### Streamable-HTTP Transport (Recommended for production deployments)
Configure your client's `mcp.json` to launch the Streamable-HTTP server via `uvx` with explicit host and port definition:

```json
{
  "mcpServers": {
    "github-agent": {
      "command": "uvx",
      "args": [
        "--from",
        "github-agent",
        "github-mcp"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "GITHUB_URL": "your_github_url_here",
        "GITHUB_VERIFY": "your_github_verify_here",
        "DEBUG": "your_debug_here",
        "PYTHONUNBUFFERED": "your_pythonunbuffered_here",
        "GITHUB_TOKEN": "your_github_token_here"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed remote or local Streamable-HTTP instance:

```json
{
  "mcpServers": {
    "github-agent": {
      "url": "http://localhost:8000/github-agent/mcp"
    }
  }
}
```

Deploying the Streamable-HTTP server via Docker:

```bash
docker run -d \
  --name github-agent-mcp \
  -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e GITHUB_URL="your_value" \
  -e GITHUB_VERIFY="your_value" \
  -e DEBUG="your_value" \
  -e PYTHONUNBUFFERED="your_value" \
  -e GITHUB_TOKEN="your_value" \
  knucklessg1/github-agent:latest
```

---

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
    image: knucklessg1/github-agent:latest
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

Install the Python package locally:

```bash
# Using uv (highly recommended)
uv pip install github-agent[all]

# Using standard pip
python -m pip install github-agent[all]
```

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
