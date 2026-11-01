# Contract: delivery classes

Provider: medulla MEDULLA-SDLC-R019.1
Consumer: github-agent CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.14

The engine declares the delivery classes. The connector references them in its mapping and never redeclares them.

## Wire shape

- `schema_mappings` entries map `PullRequest`, `Review`, `ReviewThread` and `CheckRun` records to the engine classes `:PullRequest`, `:Review`, `:ReviewThread` and `:CheckRun`.

## Refusal codes

- `PACK_CLASS_UNROOTED`: a mapped class does not reach an engine core class.
- A null `ontology_class` fails the connector certification.

## Authoritative types

- Engine vocabulary: `crates/medulla-core/ontology/sdlc_lifecycle-v1.ttl` in medulla.
- Connector mapping: `github_agent/ontology/mappings/source.yaml`.

## Test vectors

- `tests/test_cq_paths.py::test_delivery_records_map_to_shared_classes`.
