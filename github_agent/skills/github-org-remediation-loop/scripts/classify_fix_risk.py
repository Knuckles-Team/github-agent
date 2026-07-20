#!/usr/bin/env python3
"""Deterministic fix-risk classifier for freshly-authored remediation diffs.

`github-triage-resolver`'s `classify_safe.py` answers "is this *existing* PR/issue
safe to auto-resolve" (mergeable state, checks, allow-class). This script answers a
different question that only makes sense AFTER `sdd-implementer` has produced a
diff: "is the fix **we just wrote** safe to auto-merge, or does it need a human to
look at it first." The two are complementary, not overlapping — this loop runs
`classify_safe.py` for the existing-item leg (Loop B's incoming PRs, and Loop A's
"is this issue already fixed") and this script for the diff-risk leg (Loop A
always, Loop B only when it re-implements per step 5b).

Verdict:
  low_risk       ALL of:
                   - verifier_pass is true (spec-verifier CHECKLIST.md 100% green)
                   - files_changed <= --max-files (default 5)
                   - lines_changed <= --max-lines (default 150)
                   - no touched path matches an infra-surface pattern
                   - checks_state == "success" on the fix branch
                   - evidence_confirmed is true (issue: a regression test
                     reproduces the original report and now passes; CI: the
                     specific failing job passes on rerun)
  elevated_risk  otherwise — the reason lists every gate that failed.

Infra-surface patterns (never low_risk, regardless of diff size): migrations,
schema, auth, secrets, .github/workflows, bumpversion config, Dockerfile/compose.
Override with --infra-patterns for a repo with different conventions.

This script makes NO network calls and writes nothing. It only decides. Pure
stdlib JSON transform, same shape/spirit as classify_safe.py.

Input fields per item (extra fields ignored):
  repo, number          identifiers for the report
  files_changed         int  — count of files touched by the diff
  lines_changed         int  — additions + deletions
  touched_paths         list[str]  — repo-relative paths touched by the diff
  verifier_pass         bool — spec-verifier CHECKLIST.md is 100% green
  checks_state          "success" | "failure" | "pending" | None
  evidence_confirmed    bool — regression test / CI rerun confirms the fix

Usage:
  python classify_fix_risk.py items.json --format md
  cat item.json | python classify_fix_risk.py --max-files 5 --max-lines 150 --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

DEFAULT_INFRA_PATTERNS = [
    "migrations/",
    "schema/",
    "auth/",
    "secrets/",
    ".github/workflows/",
    ".bumpversion.cfg",
    "bumpversion",
    "Dockerfile",
    "compose.yml",
    "docker-compose",
]


def _items(blob: Any) -> list[dict]:
    if isinstance(blob, list):
        return [i for i in blob if isinstance(i, dict)]
    if isinstance(blob, dict):
        for k in ("data", "items"):
            if isinstance(blob.get(k), list):
                return [i for i in blob[k] if isinstance(i, dict)]
        return [blob]
    return []


def _infra_hits(touched_paths: list[str], patterns: list[str]) -> list[str]:
    hits = []
    for path in touched_paths:
        for pat in patterns:
            if pat.lower() in path.lower():
                hits.append(path)
                break
    return hits


def classify(item: dict, max_files: int, max_lines: int, patterns: list[str]) -> dict:
    repo, num = item.get("repo", "?"), item.get("number", "?")
    out = {"repo": repo, "number": num, "verdict": "elevated_risk", "reason": ""}

    files_changed = item.get("files_changed")
    lines_changed = item.get("lines_changed")
    touched_paths = item.get("touched_paths") or []
    verifier_pass = bool(item.get("verifier_pass"))
    checks_state = (item.get("checks_state") or "").lower()
    evidence_confirmed = bool(item.get("evidence_confirmed"))

    why = []
    if not verifier_pass:
        why.append("spec-verifier checklist not 100% green")
    if not isinstance(files_changed, int) or files_changed > max_files:
        why.append(f"files_changed={files_changed} exceeds max_files={max_files}")
    if not isinstance(lines_changed, int) or lines_changed > max_lines:
        why.append(f"lines_changed={lines_changed} exceeds max_lines={max_lines}")
    infra_hits = _infra_hits(touched_paths, patterns)
    if infra_hits:
        why.append(f"touches infra surface: {', '.join(infra_hits)}")
    if checks_state != "success":
        why.append(f"checks_state={checks_state or 'unknown'} (need success)")
    if not evidence_confirmed:
        why.append("fix not confirmed by a regression test / CI rerun")

    if not why:
        out.update(
            verdict="low_risk",
            reason="verifier green, small config-only diff, checks green, evidence confirmed",
        )
    else:
        out["reason"] = "; ".join(why)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*", help="item JSON file(s); omit for stdin")
    ap.add_argument("--max-files", type=int, default=5)
    ap.add_argument("--max-lines", type=int, default=150)
    ap.add_argument(
        "--infra-patterns",
        default=",".join(DEFAULT_INFRA_PATTERNS),
        help="comma list of substrings; any touched path containing one forces elevated_risk",
    )
    ap.add_argument("--format", choices=["json", "md"], default="md")
    args = ap.parse_args()
    patterns = [p.strip() for p in args.infra_patterns.split(",") if p.strip()]

    blobs: list[dict] = []
    if args.files:
        for f in args.files:
            with open(f) as fh:
                blobs.extend(_items(json.load(fh)))
    else:
        blobs.extend(_items(json.load(sys.stdin)))

    verdicts = [classify(i, args.max_files, args.max_lines, patterns) for i in blobs]
    if args.format == "json":
        print(json.dumps(verdicts, indent=2))
        return
    icon = {"low_risk": "✅ low_risk", "elevated_risk": "⚠️ elevated_risk"}
    print("| Item | Verdict | Reason |")
    print("|------|---------|--------|")
    for v in verdicts:
        print(
            f"| {v['repo']}#{v['number']} | {icon.get(v['verdict'], v['verdict'])} | {v['reason']} |"
        )
    n_low = sum(v["verdict"] == "low_risk" for v in verdicts)
    n_elev = sum(v["verdict"] == "elevated_risk" for v in verdicts)
    print(
        f"\n**{n_low} low_risk · {n_elev} elevated_risk** (of {len(verdicts)}). "
        f"low_risk may auto-merge+close under autonomy=auto-safe; elevated_risk always opens a "
        f"needs-human-review PR."
    )


if __name__ == "__main__":
    main()
