"""Characterization tests for github-tools's inspect_pr_checks.py.

Covers fetch_checks (shells out to `gh pr checks`, with a field-fallback
retry) and render_results (pure stdout formatting). Loaded via importlib
from its file path (no ``__init__.py``; standalone CLI script). fetch_checks
is exercised by monkeypatching run_gh_command -- no real `gh` invocation.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent
    / "github_agent"
    / "skills"
    / "github-tools"
    / "scripts"
    / "inspect_pr_checks.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("inspect_pr_checks", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


# ---- fetch_checks -------------------------------------------------------


def test_fetch_checks_primary_call_succeeds(mod, monkeypatch, tmp_path):
    checks = [{"name": "build", "state": "SUCCESS"}]

    def fake_run(args, cwd):
        assert "--json" in args
        return mod.GhResult(0, json.dumps(checks), "")

    monkeypatch.setattr(mod, "run_gh_command", fake_run)
    result = mod.fetch_checks("1", tmp_path)
    assert result == checks


def test_fetch_checks_falls_back_to_available_fields(mod, monkeypatch, tmp_path):
    calls = []
    fallback_checks = [{"name": "build", "state": "pass", "bucket": "pass"}]

    def fake_run(args, cwd):
        calls.append(args)
        if len(calls) == 1:
            return mod.GhResult(
                1,
                "",
                "Unknown JSON field: \"conclusion\"\n"
                "Available fields:\n"
                "  bucket\n  link\n  name\n  startedAt\n  completedAt\n"
                "  state\n  workflow",
            )
        return mod.GhResult(0, json.dumps(fallback_checks), "")

    monkeypatch.setattr(mod, "run_gh_command", fake_run)
    result = mod.fetch_checks("1", tmp_path)
    assert result == fallback_checks
    assert len(calls) == 2
    # second call used only fields the CLI reported as available
    assert "bucket" in calls[1][-1]
    assert "conclusion" not in calls[1][-1]


def test_fetch_checks_no_available_fields_returns_none(mod, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        mod, "run_gh_command", lambda args, cwd: mod.GhResult(1, "", "some other error")
    )
    result = mod.fetch_checks("1", tmp_path)
    assert result is None
    assert "some other error" in capsys.readouterr().err


def test_fetch_checks_fallback_retry_also_fails_returns_none(mod, monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(args, cwd):
        calls.append(args)
        if len(calls) == 1:
            return mod.GhResult(
                1, "", "Unknown JSON field\nAvailable fields:\n  name\n  state"
            )
        return mod.GhResult(1, "", "still broken")

    monkeypatch.setattr(mod, "run_gh_command", fake_run)
    result = mod.fetch_checks("1", tmp_path)
    assert result is None
    assert "still broken" in capsys.readouterr().err


def test_fetch_checks_invalid_json_returns_none(mod, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mod, "run_gh_command", lambda args, cwd: mod.GhResult(0, "not json", ""))
    result = mod.fetch_checks("1", tmp_path)
    assert result is None
    assert "unable to parse checks JSON" in capsys.readouterr().err


def test_fetch_checks_non_list_json_returns_none(mod, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        mod, "run_gh_command", lambda args, cwd: mod.GhResult(0, json.dumps({"a": 1}), "")
    )
    result = mod.fetch_checks("1", tmp_path)
    assert result is None
    assert "unexpected checks JSON shape" in capsys.readouterr().err


# ---- render_results -------------------------------------------------------


def test_render_results_full_check(mod, capsys):
    results = [
        {
            "name": "build",
            "detailsUrl": "https://example.com/details",
            "runId": "111",
            "jobId": "222",
            "status": "completed",
            "run": {
                "headBranch": "main",
                "headSha": "abcdef1234567890",
                "workflowName": "CI",
                "conclusion": "failure",
                "url": "https://example.com/run",
            },
            "note": "a note",
            "logSnippet": "line1\nline2",
        }
    ]
    mod.render_results("42", results)
    out = capsys.readouterr().out
    assert "PR #42: 1 failing checks analyzed." in out
    assert "Check: build" in out
    assert "Details: https://example.com/details" in out
    assert "Run ID: 111" in out
    assert "Job ID: 222" in out
    assert "Status: completed" in out
    assert "Workflow: CI (failure)" in out
    assert "Branch/SHA: main abcdef123456" in out
    assert "Run URL: https://example.com/run" in out
    assert "Note: a note" in out
    assert "Failure snippet:" in out
    assert "  line1" in out


def test_render_results_error_skips_snippet(mod, capsys):
    results = [{"name": "build", "status": "failed", "error": "log fetch timed out"}]
    mod.render_results("1", results)
    out = capsys.readouterr().out
    assert "Error fetching logs: log fetch timed out" in out
    assert "Failure snippet" not in out
    assert "No snippet available." not in out


def test_render_results_no_snippet_available(mod, capsys):
    results = [{"name": "build", "status": "failed"}]
    mod.render_results("1", results)
    out = capsys.readouterr().out
    assert "No snippet available." in out


def test_render_results_empty_list(mod, capsys):
    mod.render_results("1", [])
    out = capsys.readouterr().out
    assert "PR #1: 0 failing checks analyzed." in out
