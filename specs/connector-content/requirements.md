# CONN-GITHUB-AGENT-CONNECTOR-CONTENT — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R001` | **Default content.** `pyproject.toml` declares `github-mcp` as the single `agent_connector_sdk.connector` entry point. The server lists the packaged `skill://` and `ontology://` resources and the packaged prompts with no `content=` argument. Proving test: `tests/test_default_content_surfaces.py`. | `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. |
| `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R002` | **Ontology joins the parent graph.** Each `<package>/ontology/*.ttl` that declares classes imports `http://knuckles.team/kg`, and each class reaches an epistemic-graph parent class through `rdfs:subClassOf`. No class is pending. Proving test: `tests/test_ontology_parent_link.py`. | The test parses each ontology and finds no orphan class beyond the pending list. |
| `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003` | **Ingestible sources.** `github_agent/ontology/ingest_sources.json` declares the `issues` source (record type `Issue`, read-only tool `github_search_issues`, open issues involving the authenticated user, first 50) under the fleet name `github-mcp`. The record type is a key of the manifest `schema_mappings`. Proving test: `tests/test_ingest_sources.py`. | `load_ingest_sources` returns the declaration; the tool name, record type and read-only annotation match. |
| `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R004` | **Competency-question paths.** `github.ttl` references the shared `Repository`, `PullRequest`, `Release`, `Branch`, `Commit` and `PipelineRun` and never redeclares them, so each has the one parent chain the epistemic-graph ontology decides; `CheckRun` extends `PipelineRun`; `state` carries a domain of `Issue` and `PullRequest`; no `schema_mappings` entry has a null `ontology_class`. Proving test: `tests/test_cq_paths.py`. | The test finds no shared-class declaration, the `state` domain on `Issue` and no null mapping. |
