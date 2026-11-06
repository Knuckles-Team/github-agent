# Usage — MCP / API / CLI

`github-agent` exposes the same capability three ways: as **MCP tools** an agent
calls, as a **Python API** (`Api`) the operator import, and as **CLI** servers the operator run. The
supervisor architecture and the specialized child agents are detailed in
[Overview](overview.md).

## As an MCP server

Once [deployed](deployment.md), the server registers one typed tool per GitHub operation
(`github_get_repositories`, `github_create_issue`, `github_get_pull_requests`,
`github_merge_pull_request`, `github_get_workflow_runs`, ...), plus
`github_pulls_enable_auto_merge`, `github_pulls_disable_auto_merge`, `github_graphql`,
`github_discover_graphql_schema` and the ingestion tools. Delete, remove and other
destructive tools take `confirm=true`.

Example agent prompts that map onto these tools:

- *"List my repositories"* -> `github_get_repositories`
- *"Open an issue titled 'Bug' in `owner/repo`"* -> `github_create_issue`
- *"Search GitHub for repositories matching 'mcp server'"* -> `github_search_repositories`
- *"Show the latest workflow runs for `owner/repo`"* -> `github_get_workflow_runs`

## As a Python API

`Api` (`github_agent.api_client.Api`) is a `requests`-based REST client composed from
per-domain mixins. Build it directly, or from the environment with `get_client()`.

```python
import os

from github_agent.api_client import Api

api = Api(
    url="https://api.github.com",
    token=os.environ["GITHUB_TOKEN"],
)

# Reads
repos = api.get_repositories()  # paginated repositories
repo = api.get_repository(owner="Knuckles-Team", repo="github-agent")
issues = api.get_issues(owner="Knuckles-Team", repo="github-agent")
commits = api.get_commits(owner="Knuckles-Team", repo="github-agent")
results = api.search_repositories(query="mcp server")

for r in repos.data:
    print(r.full_name)
```

Build a client straight from the environment:

```python
from github_agent.auth import get_client

api = get_client()  # reads endpoint/credentials and AgentConfig TLS policy
```

### Writes

Write methods follow the same shape and require a token with the appropriate scopes:

```python
api.create_issue(owner="Knuckles-Team", repo="github-agent", title="New issue")
api.create_pull_request(
    owner="Knuckles-Team",
    repo="github-agent",
    title="Feature",
    head="feature-branch",
    base="main",
)
api.create_release(owner="Knuckles-Team", repo="github-agent", tag_name="v1.0.0")
```

## As a CLI

Two console scripts are installed:

```bash
# The MCP server
github-mcp --transport streamable-http --host 0.0.0.0 --port 8000

# The A2A supervisor agent
github-agent --provider openai --model-id gpt-4o
```

The agent server delegates to specialized child agents (Repos, Issues, Pull Requests,
Actions, Organizations, and more) — see [Overview](overview.md) for the full roster
and the supervisor-worker flow. Provider, model, and API key are supplied via
`--provider` / `--model-id` flags or the `PROVIDER` / `MODEL_ID` environment
variables described in [Deployment](deployment.md#agent-server-a2a).
