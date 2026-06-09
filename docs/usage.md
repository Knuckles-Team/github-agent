# Usage — MCP / API / CLI

`github-agent` exposes the same capability three ways: as **MCP tools** an agent
calls, as a **Python API** (`Api`) you import, and as **CLI** servers you run. The
supervisor architecture and the specialized child agents are detailed in
[Overview](overview.md).

## As an MCP server

Once [deployed](deployment.md), the server registers eleven action-dispatch tool
domains. Each domain takes an `action` and a JSON `params_json` payload; each is
gated by its own environment switch (all default `True`).

| Tool | Actions |
|---|---|
| `github_repos` | `list`, `get`, `create`, `update`, `delete` |
| `github_issues` | `list`, `get`, `create`, `update` |
| `github_pulls` | `list`, `get`, `create`, `update` |
| `github_contents` | `get`, `create`, `update`, `delete` |
| `github_branches` | `list`, `get`, `create`, `delete`, `get_protection`, `update_protection`, `delete_protection` |
| `github_commits` | `list`, `get` |
| `github_search` | `repositories`, `issues`, `code` |
| `github_orgs` | `repos`, `members`, `teams` |
| `github_collaborators` | `list`, `add`, `remove` |
| `github_actions` | `list_workflows`, `list_runs`, `get_run`, `trigger_dispatch`, `rerun`, `cancel`, `delete_run` |
| `github_releases` | `list`, `get`, `create`, `update`, `delete` |

Example agent prompts that map onto these tools:

- *"List my repositories"* → `github_repos` (`list`)
- *"Open an issue titled 'Bug' in `owner/repo`"* → `github_issues` (`create`)
- *"Search GitHub for repositories matching 'mcp server'"* → `github_search` (`repositories`)
- *"Show the latest workflow runs for `owner/repo`"* → `github_actions` (`list_runs`)

## As a Python API

`Api` (`github_agent.api_client.Api`) is a `requests`-based REST client composed from
per-domain mixins. Build it directly, or from the environment with `get_client()`.

```python
from github_agent.api_client import Api

api = Api(
    url="https://api.github.com",
    token="ghp_your_personal_access_token",
    verify=True,
)

# Reads
repos = api.get_repositories()                          # paginated repositories
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
api = get_client()        # reads GITHUB_URL / GITHUB_TOKEN / GITHUB_VERIFY
```

### Writes

Write methods follow the same shape and require a token with the appropriate scopes:

```python
api.create_issue(owner="Knuckles-Team", repo="github-agent", title="New issue")
api.create_pull_request(
    owner="Knuckles-Team", repo="github-agent",
    title="Feature", head="feature-branch", base="main",
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
