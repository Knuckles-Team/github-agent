# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`github-agent` supports local stdio, a loopback-only development listener, a
least-privilege stdio container, and a remote authenticated HTTPS boundary.
Provider endpoint, credential, selector, identity, and trust material are supplied
at runtime through `AgentConfig`; none is stored in this repository.

### Installed stdio process

```json
{
  "mcpServers": {
    "github": {
      "command": "github-mcp",
      "args": [],
      "env": {"MCP_TOOL_MODE": "intent"}
    }
  }
}
```

### Loopback development listener

```bash
github-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Do not expose this listener beyond loopback. Network deployments require direct TLS
or an explicitly trusted TLS-terminating ingress, configured authentication, exact
`MCP_ALLOWED_HOSTS`, and an exact trusted-proxy CIDR policy.

### Least-privilege local container

```bash
docker run -i --rm \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  -e TRANSPORT=stdio \
  registry.example.invalid/github-agent@sha256:<digest> github-mcp
```

The operator projects the selected AgentConfig profile into the process at runtime;
the image remains immutable and contains no environment connection profile.

### Remote authenticated HTTPS endpoint

```json
{
  "mcpServers": {
    "github": {"url": "https://service.example.invalid/mcp"}
  }
}
```

Store the real remote URL, outbound identity reference, and TLS-profile reference in
`AgentConfig`, not in MCP client JSON or documentation.
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
| `TLS_PROFILE` / `TLS_PROFILE_REF` | _(system trust)_ | AgentConfig private-CA/mTLS profile; verification remains mandatory |
| `HOST` | `0.0.0.0` | Bind address for HTTP transports |
| `PORT` | `8000` | Bind port for HTTP transports |
| `TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |

Each tool domain is gated by its own switch — `REPOTOOL`, `ISSUETOOL`, `PULLTOOL`,
`CONTENTTOOL`, `BRANCHTOOL`, `COMMITTOOL`, `SEARCHTOOL`, `ORGTOOL`,
`COLLABORATORTOOL`, `ACTIONTOOL`, `RELEASETOOL` (all default `True`). The full set,
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
    image: example/github-agent@sha256:<digest>
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
    image: example/github-agent@sha256:<digest>
    hostname: github-agent-mcp
    env_file: [../.env]
    environment:
      - TRANSPORT=streamable-http
      - HOST=0.0.0.0
      - PORT=8000
    ports: ["8000:8000"]

  github-agent-agent:
    image: example/github-agent@sha256:<digest>
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
# Internal (self-signed) — homelab .example.invalid zone
github-agent.example.invalid {
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
curl -s "http://technitium.example.invalid:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=github-agent.example.invalid" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=192.0.2.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `github-agent.example.invalid → <caddy-host-ip>` in the Technitium web
console (`http://technitium.example.invalid:5380`). The ecosystem
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
        "MCP_TOOL_MODE": "condensed",
        "GITHUB_URL": "https://api.github.com",
        "GITHUB_TOKEN": "<GITHUB_TOKEN>",
        "REPOTOOL": "True",
        "ISSUETOOL": "True",
        "PULLTOOL": "True"
      }
    }
  }
}
```

For a remote HTTP server, point the client at `http://github-agent.example.invalid/mcp` instead.
