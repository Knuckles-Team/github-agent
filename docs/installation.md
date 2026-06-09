# Installation

`github-agent` is a standard Python package and a prebuilt container image. Pick the
path that matches how you want to run it.

## Requirements

- **Python 3.11 – 3.14**.
- A **GitHub personal access token** (or GitHub App token) with the scopes for the
  operations you intend to perform. Reads of public data work unauthenticated, but
  most tools expect a token.

## From PyPI (recommended)

```bash
pip install github-agent
```

The base install pulls in `agent-utilities[agent,logfire]`, which provides the
FastMCP runtime, the Pydantic-AI agent stack, and Logfire/OpenTelemetry tracing —
everything needed to run both the MCP server and the A2A agent.

### Optional extras

| Extra | Install | Pulls in |
|---|---|---|
| `test` | `pip install "github-agent[test]"` | `pytest`, `pytest-asyncio`, `pytest-xdist` for the test suite |

## From source

```bash
git clone https://github.com/Knuckles-Team/github-agent.git
cd github-agent
pip install -e ".[test]"          # editable install with the test extra
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[test]"
uv run github-mcp
```

## Prebuilt Docker image

A multi-stage, slim image is published on every release (entrypoint `github-mcp`):

```bash
docker pull knucklessg1/github-agent:latest

docker run --rm -i \
  -e GITHUB_URL=https://api.github.com \
  -e GITHUB_TOKEN=ghp_your_personal_access_token \
  knucklessg1/github-agent:latest        # stdio transport (default)
```

For an HTTP server with a published port, see [Deployment](deployment.md).

## Verify the install

```bash
github-mcp --help
github-agent --help
python -c "import github_agent; print(github_agent.__file__)"
```

## Next steps

- **[Deployment](deployment.md)** — run it as a long-lived MCP and agent server behind Caddy + DNS.
- **[Usage](usage.md)** — call the tools, the `Api` client, and the CLI.
- **[Configuration](deployment.md#configuration-environment)** — every environment variable.
