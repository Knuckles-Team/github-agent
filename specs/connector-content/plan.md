# CONN-GITHUB-AGENT-CONNECTOR-CONTENT — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Architecture

`pyproject.toml` declares `github-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `github_agent` from it.
