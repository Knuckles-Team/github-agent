"""Characterization tests for the github-ci-failure-sweep skill's summarize_runs.py.

Pure JSON transform, no network/auth. Loaded via importlib from its file path
(no ``__init__.py``; standalone CLI script, same convention as
test_skill_backlog_build_plan.py).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent
    / "github_agent"
    / "skills"
    / "github-ci-failure-sweep"
    / "scripts"
    / "summarize_runs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("ci_summarize_runs", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def summarize_runs():
    return _load_module()


# ---- _extract_runs ---------------------------------------------------------


def test_extract_runs_bare_list(summarize_runs):
    blob = [{"id": 1}, "not-a-dict", {"id": 2}]
    assert summarize_runs._extract_runs(blob) == [{"id": 1}, {"id": 2}]


def test_extract_runs_mcp_result_data_key(summarize_runs):
    blob = {"status": 200, "data": [{"id": 1}]}
    assert summarize_runs._extract_runs(blob) == [{"id": 1}]


def test_extract_runs_github_native_workflow_runs_key(summarize_runs):
    blob = {"workflow_runs": [{"id": 1}]}
    assert summarize_runs._extract_runs(blob) == [{"id": 1}]


def test_extract_runs_nested_workflow_runs_under_data(summarize_runs):
    blob = {"data": {"workflow_runs": [{"id": 1}]}}
    assert summarize_runs._extract_runs(blob) == [{"id": 1}]


def test_extract_runs_unrecognized_shape_returns_empty(summarize_runs):
    assert summarize_runs._extract_runs({"nope": "nothing here"}) == []
    assert summarize_runs._extract_runs("a string") == []
    assert summarize_runs._extract_runs(None) == []


# ---- reduce_runs ------------------------------------------------------------


def _run(**kwargs):
    base = {
        "status": "completed",
        "conclusion": "failure",
        "repository": {"full_name": "acme/widgets"},
        "workflow_id": 1,
        "head_branch": "main",
        "updated_at": "2026-01-01T00:00:00Z",
        "id": 100,
        "run_number": 5,
        "event": "push",
        "html_url": "https://github.com/acme/widgets/actions/runs/100",
        "head_sha": "abcdef1234567890",
    }
    base.update(kwargs)
    return base


def test_reduce_runs_skips_incomplete_runs(summarize_runs):
    runs = [_run(status="in_progress")]
    assert summarize_runs.reduce_runs(runs) == []


def test_reduce_runs_keeps_only_failing_conclusions(summarize_runs):
    runs = [_run(conclusion="success")]
    assert summarize_runs.reduce_runs(runs) == []


def test_reduce_runs_keeps_latest_per_repo_workflow_branch(summarize_runs):
    older = _run(id=1, updated_at="2026-01-01T00:00:00Z")
    newer = _run(id=2, updated_at="2026-01-02T00:00:00Z")
    result = summarize_runs.reduce_runs([older, newer])
    assert len(result) == 1
    assert result[0]["run_id"] == 2


def test_reduce_runs_output_shape(summarize_runs):
    runs = [_run()]
    result = summarize_runs.reduce_runs(runs)
    assert result == [
        {
            "repo": "acme/widgets",
            "workflow": "workflow 1",
            "branch": "main",
            "conclusion": "failure",
            "run_id": 100,
            "run_number": 5,
            "event": "push",
            "updated_at": "2026-01-01T00:00:00Z",
            "html_url": "https://github.com/acme/widgets/actions/runs/100",
            "head_sha": "abcdef12",
        }
    ]


def test_reduce_runs_sorted_by_repo_workflow_branch(summarize_runs):
    a = _run(id=1, repository={"full_name": "b/repo"}, head_branch="x")
    b = _run(id=2, repository={"full_name": "a/repo"}, head_branch="y")
    result = summarize_runs.reduce_runs([a, b])
    assert [r["repo"] for r in result] == ["a/repo", "b/repo"]
