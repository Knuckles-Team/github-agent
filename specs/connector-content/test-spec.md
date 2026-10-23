# CONN-GITHUB-AGENT-CONNECTOR-CONTENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R001` | `tests/test_default_content_surfaces.py` | The listings include the packaged skills, ontology and prompts. |
| T-002 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R002` | `tests/test_ontology_parent_link.py` | Every declared class reaches a parent class; the import of the parent IRI is present. |
| T-003 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003` | `tests/test_ingest_sources.py` | The declaration loads; the source names a mapped record type and a read-only tool. |
| T-004 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R004` | `tests/test_cq_paths.py` | The test finds no shared-class declaration, the `state` domain on `Issue` and no null mapping. |
