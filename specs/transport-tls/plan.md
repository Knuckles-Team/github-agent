# CONN-GITHUB-AGENT-TRANSPORT-TLS — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Architecture

`github_agent/github_http.py` and `github_agent/github_gql.py` both call `resolve_tls_profile("github")`. The profile carries the per-instance CA (`GITHUB_CA_BUNDLE`, `GITHUB_CA_BUNDLE_REF`, `GITHUB_CA_DIRECTORY`, or a named `GITHUB_TLS_PROFILE`), system trust, and the minimum TLS version. The skill-script client passes `profile.ssl_context` to its connection and calls `profile.cleanup()` when the request ends.
