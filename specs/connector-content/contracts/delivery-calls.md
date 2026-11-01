# Contract: delivery calls from arbora

Provider: github-agent CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.8
Consumer: arbora ARBORA-OPS-R896.10

Arbora calls the change-proposal tools through `fleet.call`. It arms auto-merge only under an owner approval receipt (ARBORA-OPS-R896.10). It polls notifications as the intake fallback (ARBORA-OPS-R896.8, consumer of CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.12).

## Wire shape

- `github_enable_pull_request_auto_merge{owner, repo, number, expected_head_sha, merge_method, approval_ref}` returns `{armed, head_sha}`.
- `github_disable_pull_request_auto_merge{owner, repo, number}` returns `{armed: false}`.
- `github_list_notifications{since, participating, page, per_page}` returns `{items, next_since}`.

## Refusal codes

- `approval_required`: the arm call carries no approval reference.
- `head_mismatch`: the current head differs from `expected_head_sha`; nothing is armed.
- `unauthorized`, `not_found`, `deadline_exceeded`: the typed SDK refusals.

## Authoritative types

- `github_agent/api/api_client_pulls.py` and `github_agent/api/api_client_notifications.py`.

## Test vectors

- `tests/test_change_proposal_tools.py::test_arm_needs_approval_ref_and_matching_head`.
