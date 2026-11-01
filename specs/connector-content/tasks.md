# CONN-GITHUB-AGENT-CONNECTOR-CONTENT — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

Each task line reads `- [ ] T001 [P] [US1] <ID>-R012.4 description`. Order is producer first.

- [x] T001 [US1] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R001 Declare the fleet identity and serve the packaged content by default.
- [x] T002 [US1] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R002 Link the ontology classes to the medulla parent classes.
- [x] T003 [US1] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003 Declare the `issues` ingest source in the packaged `ingest_sources.json`.
- [x] T004 [US1] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R004 Reference the shared repository and CI classes, give `state` a domain and map every record type to a class.
- [ ] T005 [US1] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R005 Declare access contracts for the competency reads.
- [ ] T006 [US1] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R006 Repository file write tool over the SDK repository effect transport.
- [ ] T007 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007 Change-proposal delivery tools (rollup).
- [ ] T008 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.1 Read one pull request as a change-proposal record.
- [ ] T009 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.2 List the reviews of a pull request.
- [ ] T010 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.3 List the review threads of a pull request.
- [ ] T011 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.4 Resolve one review thread.
- [ ] T012 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.5 Reply to a review comment.
- [ ] T013 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.6 List the check runs of a commit.
- [ ] T014 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.7 Read the combined commit status of a ref.
- [ ] T015 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.8 Enable auto-merge for an expected head commit.
- [ ] T016 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.9 Disable auto-merge.
- [ ] T017 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.10 Close a pull request without merging.
- [ ] T018 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.11 Mark a draft pull request ready for review.
- [ ] T019 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.12 List notifications since a cursor.
- [ ] T020 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.13 Normalize webhook events to feedback events.
- [ ] T021 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.14 Map delivery records to the shared ontology classes.
- [ ] T022 [US2] CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.15 Hot-path proof: served change-proposal tools.
