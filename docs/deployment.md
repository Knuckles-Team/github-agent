# Deployment

This page covers running `github-agent` as a long-lived service: the transports, a
Docker Compose stack, the optional A2A agent server, putting it behind a Caddy
reverse proxy, and giving it a DNS name with Technitium.

> `github-agent` ships **two** console scripts: an **MCP server** (`github-mcp`) — a
> typed, deterministic tool surface a policy router / agent calls — and an **A2A
> agent server** (`github-agent`) that delegates GitHub work to specialized child
> agents. Deploy the MCP server alone, or both together.

## Run the MCP server

The transport is selected with `--transport` (or the `TRANSPORT` env var):

=== "stdio (default)"

    ```bash
    github-mcp
    ```
    For IDE / desktop MCP clients that launch the server as a subprocess.

=== "streamable-http"

    ```bash
    github-mcp --transport streamable-http --host 0.0.0.0 --port 8000
    ```
    A network server with a `/health` endpoint and `/mcp` route.

=== "sse"

    ```bash
    github-mcp --transport sse --host 0.0.0.0 --port 8000
    ```

Health check (HTTP transports):

```bash
curl -s http://localhost:8000/health        # {"status":"OK"}
```

## Configuration (environment)

`github-agent` is configured entirely from the environment. The **required** set for
the MCP server:

| Var | Default | Meaning |
|---|---|---|
| `GITHUB_URL` | `https://api.github.com` | GitHub REST API base URL (set for GitHub Enterprise) |
| `GITHUB_TOKEN` | _unset_ | Personal access token / GitHub App token (Bearer) |
| `GITHUB_VERIFY` | `True` | Verify TLS (set `False` only for self-signed Enterprise) |
| `HOST` | `0.0.0.0` | Bind address for HTTP transports |
| `PORT` | `8000` | Bind port for HTTP transports |
| `TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |

Each tool domain is gated by its own switch — `REPOSTOOL`, `ISSUETOOL`, `PULLSTOOL`,
`CONTENTSTOOL`, `BRANCHESTOOL`, `COMMITSTOOL`, `SEARCHTOOL`, `ORGSTOOL`,
`COLLABORATORSTOOL`, `ACTIONSTOOL`, `RELEASESTOOL` (all default `True`). The full set,
including telemetry (`ENABLE_OTEL`, `OTEL_EXPORTER_OTLP_*`) and access governance
(`EUNOMIA_TYPE`), is documented in
[`.env.example`](https://github.com/Knuckles-Team/github-agent/blob/main/.env.example).
Copy it to `.env` and populate only what you use; an absent `GITHUB_TOKEN` leaves the
client unauthenticated rather than raising.

## Backing service

GitHub is a **managed (SaaS) service** — there is no backing system to deploy. Point
`GITHUB_URL` at `https://api.github.com` for GitHub.com, or at your GitHub Enterprise
Server REST endpoint (for example, `https://github.example.com/api/v3`), and supply a
`GITHUB_TOKEN`. Only connection configuration is required.

## Docker Compose

The repo ships [`docker/mcp.compose.yml`](https://github.com/Knuckles-Team/github-agent/blob/main/docker/mcp.compose.yml).
It reads a sibling `.env` and publishes the HTTP server on `:8000`:

```yaml
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
```

```bash
cp .env.example .env          # then set GITHUB_TOKEN and GITHUB_URL
docker compose -f docker/mcp.compose.yml up -d
docker compose -f docker/mcp.compose.yml logs -f
```

## Agent server (A2A)

The A2A supervisor agent is published as the `github-agent` console script and
[`docker/agent.compose.yml`](https://github.com/Knuckles-Team/github-agent/blob/main/docker/agent.compose.yml),
which provisions the MCP server **and** the agent together. The agent reaches the MCP
server over `MCP_URL` and publishes its own HTTP API (and optional web UI) on `:9016`:

```yaml
services:
  github-agent-mcp:
    image: knucklessg1/github-agent:latest
    hostname: github-agent-mcp
    env_file: [../.env]
    environment:
      - TRANSPORT=streamable-http
      - HOST=0.0.0.0
      - PORT=8000
    ports: ["8000:8000"]

  github-agent-agent:
    image: knucklessg1/github-agent:latest
    depends_on: [github-agent-mcp]
    command: ["github-agent"]
    env_file: [../.env]
    environment:
      - HOST=0.0.0.0
      - PORT=9016
      - MCP_URL=http://github-agent-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
    ports: ["9016:9016"]
```

```bash
docker compose -f docker/agent.compose.yml up -d
curl -s http://localhost:9016/health        # agent health
```

Configure the agent's model with `PROVIDER`, `MODEL_ID`, and the matching provider
API key (for example `LLM_API_KEY` / `OPENAI_API_KEY`).

## Behind a Caddy reverse proxy

Expose the HTTP server on a hostname with automatic TLS. Add to your `Caddyfile`:

```caddy
# Internal (self-signed) — homelab .arpa zone
github-agent.arpa {
    tls internal
    reverse_proxy github-agent-mcp:8000
}
```

```caddy
# Public — automatic Let's Encrypt
github-agent.example.com {
    reverse_proxy github-agent-mcp:8000
}
```

Reload Caddy:

```bash
docker compose -f services/caddy/compose.yml exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## DNS with Technitium

Point the hostname at the host running Caddy. Via the Technitium API:

```bash
curl -s "http://technitium.arpa:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=github-agent.arpa" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=10.0.0.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `github-agent.arpa → <caddy-host-ip>` in the Technitium web
console (`http://technitium.arpa:5380`). The ecosystem
[`technitium-dns-mcp`](https://knuckles-team.github.io/technitium-dns-mcp/) automates
this as a tool.

## Register with an MCP client

Add to your client's `mcp_config.json` (multiplexer nickname `gh`):

```json
{
  "mcpServers": {
    "github-agent": {
      "command": "uv",
      "args": ["run", "github-mcp"],
      "env": {
        "GITHUB_URL": "https://api.github.com",
        "GITHUB_TOKEN": "ghp_your_personal_access_token",
        "REPOSTOOL": "True",
        "ISSUETOOL": "True",
        "PULLSTOOL": "True"
      }
    }
  }
}
```

For a remote HTTP server, point the client at `http://github-agent.arpa/mcp` instead.
