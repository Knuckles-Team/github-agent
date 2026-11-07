# CONN-GITHUB-AGENT-CONNECTOR-CONTENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R001` | `tests/test_default_content_surfaces.py` | The listings include the packaged skills, ontology and prompts. |
| T-002 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R002` | `tests/test_ontology_parent_link.py` | Every declared class reaches a parent class; the import of the parent IRI is present. |
| T-003 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R003` | `tests/test_ingest_sources.py` | The declaration loads; the source names a mapped record type and a read-only tool. |
| T-004 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R004` | `tests/test_cq_paths.py` | The test finds no shared-class declaration, the `state` domain on `Issue` and no null mapping. |
| T-005 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R005` | `tests/test_access_contracts.py::test_competency_contracts_certified` | Each recording validates against its access contract. |
| T-006 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R006` | `tests/test_repository_effect_served.py::test_tool_applies_one_conditional_file_effect` | One conditional file effect with the trailer. |
| T-007 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007` | `tests/test_change_proposal_served.py::test_served_tools_and_normalizer_round_trip` | Change-proposal delivery tools (rollup). |
| T-008 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.1` | `tests/test_change_proposal_tools.py::test_get_pull_request_maps_to_change_proposal_ref` | Read one pull request as a change-proposal record. |
| T-009 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.2` | `tests/test_change_proposal_tools.py::test_list_reviews_maps_reviewer_kind_and_commit` | List the reviews of a pull request. |
| T-010 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.3` | `tests/test_change_proposal_tools.py::test_review_threads_carry_resolved_state_and_cursor` | List the review threads of a pull request. |
| T-011 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.4` | `tests/test_change_proposal_tools.py::test_resolve_review_thread_is_idempotent` | Resolve one review thread. |
| T-012 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.5` | `tests/test_change_proposal_tools.py::test_review_reply_is_one_write_tool_and_bounds_the_body` | Reply to a review comment. |
| T-013 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.6` | `tests/test_change_proposal_tools.py::test_check_runs_map_status_and_conclusion` | List the check runs of a commit. |
| T-014 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.7` | `tests/test_change_proposal_tools.py::test_combined_status_maps_each_context` | Read the combined commit status of a ref. |
| T-015 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.8` | `tests/test_change_proposal_tools.py::test_arm_needs_approval_ref_and_matching_head` | Enable auto-merge for an expected head commit. |
| T-016 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.9` | `tests/test_change_proposal_tools.py::test_disarm_is_idempotent_and_needs_no_approval` | Disable auto-merge. |
| T-017 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.10` | `tests/test_change_proposal_tools.py::test_close_pull_request_is_destructive_and_idempotent` | Close a pull request without merging. |
| T-018 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.11` | `tests/test_change_proposal_tools.py::test_mark_ready_is_idempotent` | Mark a draft pull request ready for review. |
| T-019 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.12` | `tests/test_change_proposal_tools.py::test_notifications_page_and_return_the_next_cursor` | List notifications since a cursor. |
| T-020 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.13` | `tests/test_scm_events.py::test_each_event_maps_to_one_feedback_kind` | Normalize webhook events to feedback events. |
| T-021 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.14` | `tests/test_cq_paths.py::test_delivery_records_map_to_shared_classes` | Map delivery records to the shared ontology classes. |
| T-022 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R007.15` | `tests/test_change_proposal_served.py::test_served_tools_and_normalizer_round_trip` | Hot-path proof: served change-proposal tools. |
| T-023 | `CONN-GITHUB-AGENT-CONNECTOR-CONTENT-R008` | `tests/test_side_effect_audit.py::test_injected_text_causes_no_write_beyond_the_intended_issue` | Side-effect audit: an agent run writes only what its task asked for. |
