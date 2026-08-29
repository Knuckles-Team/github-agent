"""Characterization tests for github-org-remediation-loop's classify_fix_risk.py.

``classify`` is a safety gate: it decides whether a freshly-authored
remediation diff may auto-merge (``low_risk``) or must go to a human
(``elevated_risk``). Pure stdlib JSON transform, no network/auth. Loaded via
importlib from its file path (no ``__init__.py``; standalone CLI script).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent
    / "github_agent"
    / "skills"
    / "github-org-remediation-loop"
    / "scripts"
    / "classify_fix_risk.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("classify_fix_risk", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def classify_fix_risk():
    return _load_module()


def _clean_item(**overrides):
    item = {
        "repo": "acme/widgets",
        "number": 42,
        "files_changed": 2,
        "lines_changed": 10,
        "touched_paths": ["src/foo.py"],
        "verifier_pass": True,
        "checks_state": "success",
        "evidence_confirmed": True,
    }
    item.update(overrides)
    return item


# ---- _items -----------------------------------------------------------------


def test_items_bare_list(classify_fix_risk):
    assert classify_fix_risk._items([{"a": 1}, "skip", {"b": 2}]) == [
        {"a": 1},
        {"b": 2},
    ]


def test_items_wrapped_under_data_key(classify_fix_risk):
    assert classify_fix_risk._items({"data": [{"a": 1}]}) == [{"a": 1}]


def test_items_wrapped_under_items_key(classify_fix_risk):
    assert classify_fix_risk._items({"items": [{"a": 1}]}) == [{"a": 1}]


def test_items_bare_dict_treated_as_single_item(classify_fix_risk):
    assert classify_fix_risk._items({"repo": "x"}) == [{"repo": "x"}]


def test_items_unrecognized_type_returns_empty(classify_fix_risk):
    assert classify_fix_risk._items("not json-ish") == []
    assert classify_fix_risk._items(None) == []


# ---- classify: known-good (low_risk) — plant-good-input --------------------


def test_classify_all_clear_is_low_risk(classify_fix_risk):
    result = classify_fix_risk.classify(_clean_item(), max_files=5, max_lines=150, patterns=[])
    assert result["verdict"] == "low_risk"
    assert result["repo"] == "acme/widgets"
    assert result["number"] == 42


# ---- classify: known-bad inputs -> elevated_risk (plant-bad-input) ---------


def test_classify_verifier_not_passed_is_elevated(classify_fix_risk):
    result = classify_fix_risk.classify(
        _clean_item(verifier_pass=False), max_files=5, max_lines=150, patterns=[]
    )
    assert result["verdict"] == "elevated_risk"
    assert "spec-verifier checklist" in result["reason"]


def test_classify_too_many_files_is_elevated(classify_fix_risk):
    result = classify_fix_risk.classify(
        _clean_item(files_changed=100), max_files=5, max_lines=150, patterns=[]
    )
    assert result["verdict"] == "elevated_risk"
    assert "files_changed=100" in result["reason"]


def test_classify_too_many_lines_is_elevated(classify_fix_risk):
    result = classify_fix_risk.classify(
        _clean_item(lines_changed=9999), max_files=5, max_lines=150, patterns=[]
    )
    assert result["verdict"] == "elevated_risk"
    assert "lines_changed=9999" in result["reason"]


def test_classify_infra_surface_touch_is_elevated(classify_fix_risk):
    result = classify_fix_risk.classify(
        _clean_item(touched_paths=[".github/workflows/ci.yml"]),
        max_files=5,
        max_lines=150,
        patterns=classify_fix_risk.DEFAULT_INFRA_PATTERNS,
    )
    assert result["verdict"] == "elevated_risk"
    assert "infra surface" in result["reason"]
    assert ".github/workflows/ci.yml" in result["reason"]


def test_classify_checks_not_success_is_elevated(classify_fix_risk):
    result = classify_fix_risk.classify(
        _clean_item(checks_state="failure"), max_files=5, max_lines=150, patterns=[]
    )
    assert result["verdict"] == "elevated_risk"
    assert "checks_state=failure" in result["reason"]


def test_classify_evidence_not_confirmed_is_elevated(classify_fix_risk):
    result = classify_fix_risk.classify(
        _clean_item(evidence_confirmed=False), max_files=5, max_lines=150, patterns=[]
    )
    assert result["verdict"] == "elevated_risk"
    assert "not confirmed" in result["reason"]


def test_classify_multiple_failures_all_listed(classify_fix_risk):
    result = classify_fix_risk.classify(
        _clean_item(verifier_pass=False, checks_state="pending"),
        max_files=5,
        max_lines=150,
        patterns=[],
    )
    assert "spec-verifier checklist" in result["reason"]
    assert "checks_state=pending" in result["reason"]


# ---- remove the plant, confirm clean again (plant-bad-input -> remove -> PASS) --


def test_classify_removing_the_bad_input_restores_low_risk(classify_fix_risk):
    bad = _clean_item(verifier_pass=False)
    assert classify_fix_risk.classify(bad, 5, 150, [])["verdict"] == "elevated_risk"
    good = _clean_item(verifier_pass=True)
    assert classify_fix_risk.classify(good, 5, 150, [])["verdict"] == "low_risk"
