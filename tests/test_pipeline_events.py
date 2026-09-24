"""EH-410: GitHub Actions status changes become :PipelineRunEvent nodes by polling."""

from __future__ import annotations

import datetime as dt
from typing import Any

import github_agent.kg_ingest as kg


def test_each_status_change_after_the_cursor_is_one_idempotent_event(monkeypatch):
    written: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def capture(entities, relationships=None, **_kwargs):
        written.update({entity["id"]: entity for entity in entities})
        edges.extend(relationships or [])
        return {"nodes": len(entities), "edges": len(relationships or [])}

    monkeypatch.setattr(kg, "ingest_entities", capture)
    runs = [
        {
            "id": 9,
            "status": "in_progress",
            "conclusion": None,
            "updated_at": "2026-09-24T10:00:05Z",
        },
        {
            "id": 8,
            "status": "completed",
            "conclusion": "failure",
            "updated_at": "2026-09-24T09:00:00Z",
        },
        {
            "id": 7,
            "status": "completed",
            "conclusion": "success",
            "updated_at": dt.datetime(2026, 9, 24, 11, 0, tzinfo=dt.UTC),
        },
    ]
    res = kg.ingest_pipeline_run_events(
        runs, repo_full_name="org/repo", since="2026-09-24T09:30:00Z"
    )
    events = {k: v for k, v in written.items() if v["node_type"] == "PipelineRunEvent"}
    assert len(events) == 2, "run 8 is older than the cursor"
    assert res["cursor"] == "2026-09-24T11:00:00+00:00"
    assert {e["runId"] for e in events.values()} == {"9", "7"}
    assert {e["relationship"] for e in edges} == {"pipelineEventOf"}
    assert "github:pipelinerun:org/repo:9" in written

    again = kg.ingest_pipeline_run_events(
        runs, repo_full_name="org/repo", since=res["cursor"]
    )
    assert again == {"nodes": 0, "edges": 0, "cursor": res["cursor"]}
