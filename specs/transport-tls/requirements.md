# CONN-GITHUB-AGENT-TRANSPORT-TLS — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001` | **SDK TLS profile.** `github_agent/github_http.py` takes its SSL context from `resolve_tls_profile("github")`, so a per-instance CA kept in the profile is trusted and verification cannot be disabled. The package builds no SSL context of its own and never sets `verify=False` or silences TLS warnings. Proving test: `tests/test_github_http.py::test_request_uses_sdk_tls_profile_context`. | The connection receives the profile's context, the profile is cleaned up, and no `ssl.create_default_context` call remains in `github_agent/`. |
