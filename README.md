# GitHub Agent - A2A | AG-UI | MCP

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

*Version: 0.11.1*

## Overview

**GitHub Agent** is a powerful **Model Context Protocol (MCP)** server and **Agent-to-Agent (A2A)** system designed to interact with GitHub.

It acts as a **Supervisor Agent**, delegating tasks to a suite of specialized **Child Agents**, each focused on a specific domain of the GitHub API (e.g., Issues, Pull Requests, Repositories, Actions). This architecture allows for precise and efficient handling of complex GitHub operations.

This repository is actively maintained - Contributions are welcome!

### Capabilities:
- **Supervisor-Worker Architecture**: Orchestrates specialized agents for optimal task execution.
- **Comprehensive GitHub Coverage**: specialized agents for Issues, PRs, Repos, Actions, Organizations, and more.
- **MCP Support**: Fully compatible with the Model Context Protocol.
- **A2A Integration**: Ready for Agent-to-Agent communication.
- **Flexible Deployment**: Run via Docker, Docker Compose, or locally.

## Architecture

### System components

```mermaid
---
config:
  layout: dagre
---
flowchart TB
 subgraph subGraph0["Agent Capabilities"]
        Supervisor["Supervisor Agent"]
        Server["A2A Server - Uvicorn/FastAPI"]
        ChildAgents["Child Agents (Specialists)"]
        MCP["GitHub MCP Tools"]
  end
    Supervisor --> ChildAgents
    ChildAgents --> MCP
    User["User Query"] --> Server
    Server --> Supervisor
    MCP --> GitHubAPI["GitHub API"]

     Supervisor:::agent
     ChildAgents:::agent
     Server:::server
     User:::server
    classDef server fill:#f9f,stroke:#333
    classDef agent fill:#bbf,stroke:#333,stroke-width:2px
    style Server stroke:#000000,fill:#FFD600
    style MCP stroke:#000000,fill:#BBDEFB
    style GitHubAPI fill:#E6E6FA
    style User fill:#C8E6C9
    style subGraph0 fill:#FFF9C4
```

### Component Interaction

```mermaid
sequenceDiagram
    participant User
    participant Server as A2A Server
    participant Supervisor as Supervisor Agent
    participant Child as Child Agent (e.g. Issues)
    participant MCP as GitHub MCP Tools
    participant GitHub as GitHub API

    User->>Server: "Create an issue in repo X"
    Server->>Supervisor: Invoke Supervisor
    Supervisor->>Supervisor: Analyze Request & Select Specialist
    Supervisor->>Child: Delegate to Issues Agent
    Child->>MCP: Call create_issue Tool
    MCP->>GitHub: POST /repos/user/repo/issues
    GitHub-->>MCP: Issue Created JSON
    MCP-->>Child: Tool Response
    Child-->>Supervisor: Task Complete
    Supervisor-->>Server: Final Response
    Server-->>User: "Issue #123 created successfully"
```

## Specialized Agents

The Supervisor delegates tasks to these specialized agents:

| Agent Name | Description |
|:-----------|:------------|
| `GitHub_Context_Agent` | Provides context about the current user and GitHub status. |
| `GitHub_Actions_Agent` | Manages GitHub Actions workflows and runs. |
| `GitHub_Code_Security_Agent` | Handles code security scanning and alerts. |
| `GitHub_Dependabot_Agent` | Manages Dependabot alerts and configurations. |
| `GitHub_Discussions_Agent` | Manages repository discussions. |
| `GitHub_Gists_Agent` | Manages GitHub Gists. |
| `GitHub_Git_Agent` | Performs low-level Git operations (refs, trees, blobs). |
| `GitHub_Issues_Agent` | Manages Issues (create, list, update, comment). |
| `GitHub_Labels_Agent` | Manages repository labels. |
| `GitHub_Notifications_Agent` | Checks and manages notifications. |
| `GitHub_Organizations_Agent` | Manages Organization memberships and settings. |
| `GitHub_Projects_Agent` | Manages GitHub Projects (V2). |
| `GitHub_Pull_Requests_Agent` | Manages Pull Requests (create, review, merge). |
| `GitHub_Repos_Agent` | Manages Repositories (create, list, delete, settings). |
| `GitHub_Secret_Protection_Agent` | Manages secret scanning protection. |
| `GitHub_Security_Advisories_Agent` | Accesses security advisories. |
| `GitHub_Stargazers_Agent` | Views repository stargazers. |
| `GitHub_Users_Agent` | Accesses public user information. |
| `GitHub_Copilot_Agent` | Assists with coding tasks via Copilot. |
| `GitHub_Support_Docs_Agent` | Searches GitHub Support documentation. |

## Usage

### Prerequisites
- Python 3.10+
- A valid GitHub Personal Access Token (PAT) with appropriate permissions.

### Installation

```bash
pip install github-agent
```
Or using UV:
```bash
uv pip install github-agent
```

### CLI

The `github-agent` command starts the server.

| Argument | Description | Default |
|:---|:---|:---|
| `--host` | Host to bind the server to | `0.0.0.0` |
| `--port` | Port to bind the server to | `9000` |
| `--mcp-config` | Path to MCP configuration file | `mcp_config.json` |
| `--provider` | LLM Provider (openai, anthropic, google, etc.) | `openai` |
| `--model-id` | LLM Model ID | `nvidia/nemotron-3-super` |

### Running the Agent Server

```bash
github-agent --provider openai --model-id gpt-4o --api-key sk-...
```

## Security & Governance

This project is built on [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities), inheriting enterprise-grade security and governance features.

### Authentication & Authorization
| Feature | Description |
|---------|-------------|
| **OIDC Token Delegation** | RFC 8693 token exchange for user-context propagation from A2A → MCP |
| **Eunomia Policies** | Fine-grained, policy-driven tool authorization (`none`, `embedded`, `remote`) |
| **Scoped Credentials** | Tools execute with the caller's scoped identity where possible |
| **3LO / OAuth / API Token** | Multiple auth strategies with graceful fallback |

### Eunomia Policy Enforcement
Eunomia provides a policy enforcement point for all tool calls:
- **Embedded mode**: Load local `mcp_policies.json` for role-based access, sensitivity gating, and audit logging
- **Remote mode**: Forward authorization decisions to a central Eunomia policy server for multi-agent governance
- Enable via CLI: `--eunomia-type embedded --eunomia-policy-file mcp_policies.json`

### Runtime Protections
| Protection | Description |
|------------|-------------|
| **Tool Guard** | Sensitivity detection with human-in-the-loop approval gating |
| **Prompt Injection Defense** | Input scanning and repetition/loop guards |
| **Content Filtering** | Output schema enforcement and cost budget controls |
| **Stuck Loop Detection** | Automatic detection and recovery from agent loops |
| **Context Limit Warnings** | Proactive alerts before context window exhaustion |

### Graph Agent Architecture
The A2A agent uses `pydantic-graph` orchestration with:
- **RouterNode**: Lightweight classifier that routes queries to specialized domains
- **DomainNode**: Focused executor with only relevant tools loaded, preventing tool hallucination
- **Approval Gates**: Policy-driven approval workflows before sensitive operations
- **Usage Guards**: Budget and rate limiting enforcement

> **Production Recommendation**: Enable `--eunomia-type embedded` (or `remote`) + OIDC delegation + containerized deployment. See [`agent-utilities` documentation](https://github.com/Knuckles-Team/agent-utilities) for full policy configuration.

## Docker

### Build

```bash
docker build -t github-agent .
```

### Run using Docker

```bash
docker run -d \
  -p 9000:9000 \
  -e LLM_API_KEY=sk-... \
  -e MCP_CONFIG=/app/mcp_config.json \
  knucklessg1/github-agent:latest
```

### Run using Docker Compose

Create a `docker-compose.yml`:

```yaml
services:
  github-agent:
    image: knucklessg1/github-agent:latest
    ports:
      - "9000:9000"
    environment:
      - PROVIDER=openai
      - MODEL_ID=gpt-4o
      - LLM_API_KEY=${LLM_API_KEY}
    volumes:
      - ./mcp_config.json:/app/mcp_config.json
```

Then run:
```bash
docker-compose up -d
```

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)


## MCP Configuration Examples

### stdio (recommended for local development)
```json
{
  "mcpServers": {
    "github": {
      "command": ".venv/bin/github-mcp",
      "args": [],
      "env": {
        "GITHUB_TOKEN": ""
}
    }
  }
}
```

### Streamable HTTP (recommended for production)
```json
{
  "mcpServers": {
    "github": {
      "url": "http://localhost:8080/github-mcp/mcp"
    }
  }
}
```
