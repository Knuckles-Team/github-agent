"""Characterization tests for github-triage-resolver's classify_safe.py.

``classify`` is the safety gate that decides whether a PR/issue may be
auto-merged (``safe_merge``), auto-closed (``safe_close``), or must go to a
human (``skip``). Highest-cyclomatic function in the repo (31/29 pre-
refactor), so this is the most heavily plant-bad-input-tested file in the
lane. Pure stdlib JSON transform, no network/auth. Loaded via importlib from
its file path (no ``__init__.py``; standalone CLI script).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent
    / "github_agent"
    / "skills"
    / "github-triage-resolver"
    / "scripts"
    / "classify_safe.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("classify_safe", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def classify_safe():
    return _load_module()


DEFAULT_CLASSES = {"dependabot-patch", "dependabot-minor"}


def _pr(**overrides):
    item = {
        "type": "pr",
        "repo": "acme/widgets",
        "number": 1,
        "title": "Bump foo from 1.2.3 to 1.2.4",
        "draft": False,
        "mergeable_state": "clean",
        "checks_state": "success",
        "author": "dependabot[bot]",
    }
    item.update(overrides)
    return item


def _issue(**overrides):
    item = {"type": "issue", "repo": "acme/widgets", "number": 2}
    item.update(overrides)
    return item


def _classify(classify_safe, item, stale_days=60, allow_major=False, classes=None):
    return classify_safe.classify(
        item, stale_days, allow_major, classes if classes is not None else DEFAULT_CLASSES
    )


# ---- _items -----------------------------------------------------------------


def test_items_bare_list(classify_safe):
    assert classify_safe._items([{"a": 1}, "skip"]) == [{"a": 1}]


def test_items_wrapped_under_data_or_items_key(classify_safe):
    assert classify_safe._items({"data": [{"a": 1}]}) == [{"a": 1}]
    assert classify_safe._items({"items": [{"a": 1}]}) == [{"a": 1}]


def test_items_bare_dict_is_single_item(classify_safe):
    assert classify_safe._items({"repo": "x"}) == [{"repo": "x"}]


def test_items_unrecognized_type_returns_empty(classify_safe):
    assert classify_safe._items("nope") == []
    assert classify_safe._items(None) == []


# ---- _is_pr / _author / _bump_level -----------------------------------------


def test_is_pr_uses_explicit_type(classify_safe):
    assert classify_safe._is_pr({"type": "pr"}) is True
    assert classify_safe._is_pr({"type": "issue"}) is False


def test_is_pr_infers_from_shape_when_type_absent(classify_safe):
    assert classify_safe._is_pr({"head": {"ref": "x"}}) is True
    assert classify_safe._is_pr({}) is False


def test_bump_level_patch_minor_major(classify_safe):
    assert classify_safe._bump_level("Bump foo from 1.2.3 to 1.2.4") == "patch"
    assert classify_safe._bump_level("Bump foo from 1.2.3 to 1.3.0") == "minor"
    assert classify_safe._bump_level("Bump foo from 1.2.3 to 2.0.0") == "major"


def test_bump_level_non_bump_title_is_none(classify_safe):
    assert classify_safe._bump_level("Fix the bug") is None


# ---- classify: ISSUE branch --------------------------------------------------


def test_classify_issue_with_evidence_is_safe_close(classify_safe):
    result = _classify(classify_safe, _issue(resolved_evidence="fixed by #123"))
    assert result["verdict"] == "safe_close"
    assert result["action"] == "close"
    assert "fixed by #123" in result["reason"]


def test_classify_issue_without_evidence_is_skip(classify_safe):
    result = _classify(classify_safe, _issue())
    assert result["verdict"] == "skip"
    assert result["action"] == "none"
    assert "not verified resolved" in result["reason"]


# ---- classify: PR — plant-good-input (known-safe) ---------------------------


def test_classify_dependabot_patch_bump_clean_green_is_safe_merge(classify_safe):
    result = _classify(classify_safe, _pr())
    assert result["verdict"] == "safe_merge"
    assert result["action"] == "merge"


def test_classify_dependabot_minor_bump_is_safe_merge(classify_safe):
    result = _classify(classify_safe, _pr(title="Bump foo from 1.2.3 to 1.3.0"))
    assert result["verdict"] == "safe_merge"


def test_classify_human_approved_pr_is_safe_merge(classify_safe):
    result = _classify(
        classify_safe,
        _pr(author="octocat", title="Fix the bug", allow_class="approved"),
    )
    assert result["verdict"] == "safe_merge"


# ---- classify: PR — plant-bad-input (must NOT auto-merge) -------------------


def test_classify_draft_pr_is_skip(classify_safe):
    result = _classify(classify_safe, _pr(draft=True))
    assert result["verdict"] == "skip"
    assert result["reason"] == "draft PR"


def test_classify_major_bump_without_allow_major_is_skip(classify_safe):
    result = _classify(classify_safe, _pr(title="Bump foo from 1.2.3 to 2.0.0"))
    assert result["verdict"] == "skip"
    assert "not in auto-merge allow-class" in result["reason"]


def test_classify_major_bump_with_allow_major_is_safe_merge(classify_safe):
    # allow_major alone is not enough -- 'dependabot-major' must ALSO be in
    # the registered allow-classes (the default allow-classes never include
    # it, matching --allow-classes' own default of patch/minor only).
    result = _classify(
        classify_safe,
        _pr(title="Bump foo from 1.2.3 to 2.0.0"),
        allow_major=True,
        classes={"dependabot-major"},
    )
    assert result["verdict"] == "safe_merge"


def test_classify_major_bump_allow_major_but_class_not_registered_is_skip(classify_safe):
    result = _classify(
        classify_safe,
        _pr(title="Bump foo from 1.2.3 to 2.0.0"),
        allow_major=True,
        classes=DEFAULT_CLASSES,  # only patch/minor registered
    )
    assert result["verdict"] == "skip"


def test_classify_dirty_mergeable_state_is_never_safe_merge(classify_safe):
    result = _classify(classify_safe, _pr(mergeable_state="dirty"))
    assert result["verdict"] != "safe_merge"
    assert "mergeable_state=dirty" in result["reason"]


def test_classify_failing_checks_is_never_safe_merge(classify_safe):
    result = _classify(classify_safe, _pr(checks_state="failure"))
    assert result["verdict"] != "safe_merge"
    assert "checks=failure" in result["reason"]


def test_classify_class_not_registered_is_skip(classify_safe):
    # bot + patch bump, but the caller's allow-list only has 'dependabot-minor'.
    result = _classify(classify_safe, _pr(), classes={"dependabot-minor"})
    assert result["verdict"] == "skip"
    assert "not in auto-merge allow-class" in result["reason"]


def test_classify_non_bot_non_approved_author_is_skip(classify_safe):
    result = _classify(classify_safe, _pr(author="octocat", title="Fix the bug"))
    assert result["verdict"] == "skip"


# ---- classify: PR safe_close paths -------------------------------------------


def test_classify_superseded_pr_is_safe_close(classify_safe):
    result = _classify(
        classify_safe, _pr(mergeable_state="dirty", superseded_by="#456")
    )
    assert result["verdict"] == "safe_close"
    assert result["action"] == "close"
    assert "#456" in result["reason"]


def test_classify_stale_dirty_pr_is_safe_close(classify_safe):
    result = _classify(
        classify_safe, _pr(mergeable_state="dirty", age_days=90), stale_days=60
    )
    assert result["verdict"] == "safe_close"
    assert "stale 90d" in result["reason"]


def test_classify_dirty_but_not_stale_is_skip(classify_safe):
    result = _classify(
        classify_safe, _pr(mergeable_state="dirty", age_days=10), stale_days=60
    )
    assert result["verdict"] == "skip"


# ---- classify: remove the plant, confirm clean again -------------------------


def test_classify_removing_draft_plant_restores_safe_merge(classify_safe):
    bad = _classify(classify_safe, _pr(draft=True))
    assert bad["verdict"] == "skip"
    good = _classify(classify_safe, _pr(draft=False))
    assert good["verdict"] == "safe_merge"


# ---- classify: repo/number are always carried through -----------------------


def test_classify_carries_repo_and_number(classify_safe):
    result = _classify(classify_safe, _pr(repo="x/y", number=99))
    assert result["repo"] == "x/y"
    assert result["number"] == 99
