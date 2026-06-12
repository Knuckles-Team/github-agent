# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`github-agent` exposes its MCP server (console script `github-mcp`) four ways. Pick the row that
matches where the server runs relative to your MCP client, then copy the matching
`mcp_config.json` below. Replace the `<your-…>` placeholders with the values from the **Configuration / Environment Variables** section.

| # | Option | Transport | Where it runs | `mcp_config.json` key |
|---|--------|-----------|---------------|------------------------|
| 1 | stdio | `stdio` | client launches a subprocess | `command` |
| 2 | Streamable-HTTP (local) | `streamable-http` | a local network port | `command` or `url` |
| 3 | Local container / uv | `stdio` or `streamable-http` | Docker / Podman / uv on this host | `command` or `url` |
| 4 | Remote URL | `streamable-http` | a remote host behind Caddy | `url` |

### 1. stdio (local subprocess)

The client launches the server over stdio via `uvx` — best for local IDEs
(Cursor, Claude Desktop, VS Code):

```json
{
  "mcpServers": {
    "github-mcp": {
      "command": "uvx",
      "args": ["--from", "github-agent", "github-mcp"],
      "env": {
        "GITHUB_URL": "<your-github_url>"
      }
    }
  }
}
```

### 2. Streamable-HTTP (local process)

Run the server as a long-lived HTTP process:

```bash
uvx --from github-agent github-mcp --transport streamable-http --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/health        # {"status":"OK"}
```

Then either let the client launch it:

```json
{
  "mcpServers": {
    "github-mcp": {
      "command": "uvx",
      "args": ["--from", "github-agent", "github-mcp", "--transport", "streamable-http", "--port", "8000"],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "GITHUB_URL": "<your-github_url>"
      }
    }
  }
}
```

…or connect to the already-running process by URL:

```json
{
  "mcpServers": {
    "github-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

### 3. Local container / uv

**(a) Launch a container directly from `mcp_config.json`** (stdio over the container —
no ports to manage). Swap `docker` for `podman` for a daemonless runtime:

```json
{
  "mcpServers": {
    "github-mcp": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "TRANSPORT=stdio",
        "-e", "GITHUB_URL=<your-github_url>",
        "knucklessg1/github-agent:latest"
      ]
    }
  }
}
```

**(b) Run a local streamable-http container, then connect by URL:**

```bash
docker run -d --name github-mcp -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e GITHUB_URL="<your-github_url>" \
  knucklessg1/github-agent:latest
# or, from a clone of this repo:
docker compose -f docker/mcp.compose.yml up -d
```

```json
{
  "mcpServers": {
    "github-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

**(c) From a local checkout with `uv`:**

```bash
uv run github-mcp --transport streamable-http --port 8000
```

### 4. Remote URL (deployed behind Caddy)

When the server is deployed remotely (e.g. as a Docker service) and published through
Caddy on the internal `*.arpa` zone, connect with the `"url"` key — no local process or
image required:

```json
{
  "mcpServers": {
    "github-mcp": { "url": "http://github-mcp.arpa/mcp" }
  }
}
```

Caddy reverse-proxies `http://github-mcp.arpa` to the container's `:8000`
streamable-http listener; `http://github-mcp.arpa/health` returns
`{"status":"OK"}` when the service is live.
<!-- END GENERATED: deployment-options -->

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
