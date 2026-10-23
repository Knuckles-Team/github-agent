# CONN-GITHUB-AGENT-TRANSPORT-TLS — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001` | `tests/test_github_http.py::test_request_uses_sdk_tls_profile_context` | The connection uses the profile's context and the profile is cleaned up. |
| T-002 | `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001` | `tests/test_github_http.py::test_no_hand_built_ssl_context_remains` | No `create_default_context` call in `github_agent/`. |
| T-003 | `CONN-GITHUB-AGENT-TRANSPORT-TLS-R001` | `tests/test_github_http.py::test_private_ca_is_carried_through_the_profile` | A configured CA bundle appears in the resolved profile. |
