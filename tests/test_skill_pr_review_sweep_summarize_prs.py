"""Characterization tests for github-pr-review-sweep's summarize_prs.py.

Pure JSON transform, no network/auth. Loaded via importlib from its file path
(no ``__init__.py``; standalone CLI script).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent
    / "github_agent"
    / "skills"
    / "github-pr-review-sweep"
    / "scripts"
    / "summarize_prs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_prs", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def summarize_prs():
    return _load_module()


# ---- _extract ---------------------------------------------------------------


def test_extract_bare_list(summarize_prs):
    assert summarize_prs._extract([{"number": 1}, "skip"]) == [{"number": 1}]


def test_extract_mcp_result_data_key(summarize_prs):
    assert summarize_prs._extract({"data": [{"number": 1}]}) == [{"number": 1}]


def test_extract_pull_requests_key(summarize_prs):
    assert summarize_prs._extract({"pull_requests": [{"number": 1}]}) == [{"number": 1}]


def test_extract_pulls_key(summarize_prs):
    assert summarize_prs._extract({"pulls": [{"number": 1}]}) == [{"number": 1}]


def test_extract_items_key(summarize_prs):
    assert summarize_prs._extract({"items": [{"number": 1}]}) == [{"number": 1}]


def test_extract_single_pr_object(summarize_prs):
    blob = {"number": 1, "title": "fix it"}
    assert summarize_prs._extract(blob) == [blob]


def test_extract_dict_without_pr_shape_returns_empty(summarize_prs):
    assert summarize_prs._extract({"status": 200}) == []


def test_extract_unrecognized_type_returns_empty(summarize_prs):
    assert summarize_prs._extract("nope") == []
    assert summarize_prs._extract(None) == []


# ---- to_md --------------------------------------------------------------


def test_to_md_empty_rows(summarize_prs):
    assert "No open pull requests" in summarize_prs.to_md([])


def test_to_md_single_repo_single_row(summarize_prs):
    rows = [
        {
            "repo": "acme/widgets",
            "number": 7,
            "title": "Add feature",
            "author": "octocat",
            "base": "main",
            "head": "feature",
            "age_days": 3,
            "draft": False,
            "mergeable_state": "clean",
            "additions": 10,
            "deletions": 2,
        }
    ]
    md = summarize_prs.to_md(rows)
    assert "### acme/widgets" in md
    assert "| #7 | Add feature | octocat | main←feature | 3d | " in md
    assert "clean | +10/-2 |" in md
    assert "1 open PR(s)" in md


def test_to_md_groups_by_repo_and_truncates_long_title(summarize_prs):
    rows = [
        {
            "repo": "a/repo",
            "number": 1,
            "title": "x" * 60,
            "author": None,
            "base": None,
            "head": None,
            "age_days": None,
            "draft": True,
            "mergeable_state": None,
            "additions": None,
            "deletions": None,
        },
        {
            "repo": "b/repo",
            "number": 2,
            "title": "short",
            "author": "bob",
            "base": "main",
            "head": "fix",
            "age_days": 0,
            "draft": False,
            "mergeable_state": "dirty",
            "additions": 1,
            "deletions": 1,
        },
    ]
    md = summarize_prs.to_md(rows)
    assert "### a/repo" in md
    assert "### b/repo" in md
    assert "x" * 50 + "…" in md
    assert "—" in md  # missing author/base/head/age/size render as em-dash/'?'
    assert "yes |" in md  # draft
    assert "2 open PR(s)** across 2 repo(s)" in md
