# Contract: change-proposal records and the normalizer protocol

Provider: scion-sdk SCION-SCM-R003.2
Consumer: github-agent CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.13

The SDK defines the records and the normalizer protocol. The connector implements the protocol for GitHub webhooks and returns the records from its tools.

## Wire shape

- `ScmEventNormalizer.normalize(headers, body) -> ScmFeedbackEvent` after `verify_webhook` accepts the signature.
- `ScmFeedbackEvent{event_id, provider, kind, proposal, actor, actor_kind, at, body_digest}` with `kind` from the closed set of SCION-SCM-R003.1.
- Tool replies map to `ChangeProposalRef`, `ReviewRecord`, `ReviewThreadRecord` and `CheckRunRecord` (SCION-SCM-R003.1).

## Refusal codes

- `SCM_SIGNATURE_INVALID`: the signature does not match the shared secret.
- `SCM_EVENT_UNSUPPORTED`: the event type has no feedback kind.

## Authoritative types

- SDK records: `scion_sdk/scm/records.py` (SCION-SCM-R003.1).
- Connector normalizer: `github_agent/scm_events.py` (CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.13).

## Test vectors

- `tests/fixtures/scm_events/` in this repository: one recorded payload per event type.
