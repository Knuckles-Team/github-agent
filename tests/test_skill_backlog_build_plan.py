"""Characterization tests for the github-backlog-planner skill's build_plan.py.

Pure JSON -> Markdown transform, no network/auth. It lives under
``github_agent/skills/github-backlog-planner/scripts/`` with no ``__init__.py``
(skill scripts are standalone CLI tools, not an importable package), so it is
loaded here via ``importlib`` from its file path, same as running it directly.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent
    / "github_agent"
    / "skills"
    / "github-backlog-planner"
    / "scripts"
    / "build_plan.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("backlog_build_plan", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_plan():
    return _load_module()


def test_norm_fills_defaults_for_missing_fields(build_plan):
    normalized = build_plan._norm({})
    assert normalized == {
        "account": "(unknown)",
        "repo": "(unknown)",
        "kind": "issue",
        "number": None,
        "title": "(no title)",
        "url": "",
        "status": "needs-action",
        "evidence": "",
        "recommendation": "",
        "priority": "medium",
        "labels": [],
    }


def test_norm_resets_unknown_status_and_priority_to_default(build_plan):
    normalized = build_plan._norm({"status": "bogus", "priority": "urgent!!"})
    assert normalized["status"] == "needs-action"
    assert normalized["priority"] == "medium"


def test_norm_preserves_valid_fields(build_plan):
    item = {
        "account": "Knuckles-Team",
        "repo": "geniusbot",
        "kind": "PR",
        "number": 42,
        "title": "  Fix the thing  ",
        "url": "https://github.com/x/y/pull/42",
        "status": "IN-PROGRESS",
        "evidence": "  saw it  ",
        "recommendation": "  merge it  ",
        "priority": "HIGH",
        "labels": ["bug"],
    }
    normalized = build_plan._norm(item)
    assert normalized["kind"] == "pr"
    assert normalized["title"] == "Fix the thing"
    assert normalized["status"] == "in-progress"
    assert normalized["evidence"] == "saw it"
    assert normalized["recommendation"] == "merge it"
    assert normalized["priority"] == "high"
    assert normalized["labels"] == ["bug"]


def test_build_markdown_empty_items(build_plan):
    md = build_plan.build_markdown([], "Empty Plan")
    assert "# Empty Plan" in md
    assert "0 open item(s)" in md
    assert "_No outstanding items requiring action._" in md


def test_build_markdown_addressed_item_is_closable(build_plan):
    items = [
        {
            "account": "acme",
            "repo": "widgets",
            "kind": "issue",
            "number": 1,
            "title": "Fix bug",
            "url": "https://github.com/acme/widgets/issues/1",
            "status": "addressed",
            "evidence": "fixed in #2",
            "recommendation": "close it",
            "priority": "high",
        }
    ]
    md = build_plan.build_markdown(items, "Plan")
    assert "✅ Verified addressed" in md
    assert "[acme/widgets#1](https://github.com/acme/widgets/issues/1) — Fix bug" in md
    assert "_Why:_ fixed in #2" in md
    assert "_Action:_ close it" in md
    # Addressed items are excluded from the action-plan table.
    assert "_No outstanding items requiring action._" in md


def test_build_markdown_outstanding_item_grouped_into_table(build_plan):
    items = [
        {
            "account": "acme",
            "repo": "widgets",
            "kind": "pr",
            "number": 7,
            "title": "Add feature | with pipe",
            "url": "",
            "status": "needs-action",
            "priority": "low",
        }
    ]
    md = build_plan.build_markdown(items, "Plan")
    assert "### acme" in md
    assert "#### `widgets` (1)" in md
    assert "#7" in md  # no url -> plain '#7', not a markdown link
    assert "Add feature \\| with pipe" in md  # pipe escaped for the table
    assert "P3" in md  # low priority badge
    assert "🔴 Needs action" in md


def test_build_markdown_missing_recommendation_renders_em_dash(build_plan):
    items = [
        {
            "account": "acme",
            "repo": "widgets",
            "kind": "issue",
            "number": 3,
            "title": "t",
            "status": "in-progress",
        }
    ]
    md = build_plan.build_markdown(items, "Plan")
    assert "| — |" in md
