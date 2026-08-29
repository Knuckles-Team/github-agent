#!/usr/bin/env python3
"""Deterministic safety classifier for autonomous PR/issue resolution.

Given enriched item(s) — a PR or issue object (from `github_pulls action=get` /
`github_issues action=get`) augmented by the agent with `checks_state` and, for issues,
`resolved_evidence` — emit a conservative verdict:

  * safe_merge  — PR is mergeable, all checks green, and fits a low-risk allow
                  class (default: dependabot/renovate patch|minor bumps). Never
                  for major bumps, drafts, dirty/blocked merges, or failing checks.
  * safe_close  — PR: stale AND conflicted (abandoned); OR `superseded_by` given.
                  Issue: ONLY when `resolved_evidence` (a fixing commit/PR/file)
                  is supplied — an unverified issue is NEVER auto-closable.
  * skip        — anything else: leave for a human (with the reason).

This script makes NO network calls and writes nothing. It only decides. The skill
still requires explicit confirmation (or a configured autonomy level) before any
write happens. Pure stdlib JSON transform.

Input fields per item (extra fields ignored):
  type            "pr" | "issue"  (inferred from `pull_request`/`head` if absent)
  number, repo, title, draft, mergeable_state, author (login or {login}),
  checks_state    "success" | "failure" | "pending" | None   (agent supplies)
  age_days        int
  resolved_evidence  str  (issues only — e.g. "fixed by #123 / commit abc123")
  superseded_by      str  (PRs only — e.g. "#456")
  allow_class        str  (override; e.g. "approved" to permit a human-approved PR)

Usage:
  python classify_safe.py items.json --format md
  cat item.json | python classify_safe.py --stale-days 60 --format json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")
BUMP = re.compile(r"bump .+ from (\S+) to (\S+)", re.I)


def _list_of_dicts(value: Any) -> list[dict] | None:
    """``value`` filtered to its dict elements, if it is a list; else ``None``."""
    if not isinstance(value, list):
        return None
    return [i for i in value if isinstance(i, dict)]


def _items(blob: Any) -> list[dict]:
    direct = _list_of_dicts(blob)
    if direct is not None:
        return direct
    if not isinstance(blob, dict):
        return []
    for k in ("data", "items"):
        found = _list_of_dicts(blob.get(k))
        if found is not None:
            return found
    return [blob]


def _author(item: dict) -> str:
    a = item.get("author") or item.get("user") or ""
    return (a.get("login") if isinstance(a, dict) else a) or ""


def _is_pr(item: dict) -> bool:
    t = item.get("type")
    if t:
        return t == "pr"
    return bool(item.get("pull_request") or item.get("head") or item.get("base"))


def _bump_level(title: str) -> str | None:
    """patch|minor|major for a 'Bump X from a.b.c to d.e.f' title, else None."""
    m = BUMP.search(title or "")
    if not m:
        return None
    a, b = SEMVER.search(m.group(1)), SEMVER.search(m.group(2))
    if not (a and b):
        return None
    av, bv = [int(x) for x in a.groups()], [int(x) for x in b.groups()]
    if bv[0] != av[0]:
        return "major"
    if bv[1] != av[1]:
        return "minor"
    return "patch"


def _classify_issue(item: dict) -> dict:
    ev = item.get("resolved_evidence")
    if ev:
        return {
            "verdict": "safe_close",
            "action": "close",
            "reason": f"resolved — evidence: {ev}",
        }
    return {
        "verdict": "skip",
        "action": "none",
        "reason": "issue not verified resolved (no resolved_evidence) — human triage",
    }


def _pr_context(item: dict) -> dict[str, Any]:
    """Precompute the fields every PR gate below reads."""
    author = _author(item).lower()
    title = item.get("title") or ""
    bot = ("dependabot" in author) or ("renovate" in author)
    return {
        "mergeable_state": (item.get("mergeable_state") or "").lower(),
        "checks_state": (item.get("checks_state") or "").lower(),
        "author": author,
        "bot": bot,
        "level": _bump_level(title),
        "allow": item.get("allow_class") or "",
    }


def _pr_class_permitted(ctx: dict[str, Any], allow_major: bool) -> bool:
    bot, level, allow = ctx["bot"], ctx["level"], ctx["allow"]
    return (
        (bot and level in ("patch", "minor"))
        or (bot and level == "major" and allow_major)
        or (allow == "approved")
    )


def _pr_class_registered(ctx: dict[str, Any], classes: set[str]) -> bool:
    bot, level, allow = ctx["bot"], ctx["level"], ctx["allow"]
    if bot and level:
        return f"dependabot-{level}" in classes
    return allow == "approved"


def _pr_merge_class_ok(ctx: dict[str, Any], allow_major: bool, classes: set[str]) -> bool:
    return _pr_class_permitted(ctx, allow_major) and _pr_class_registered(ctx, classes)


def _pr_safe_merge_verdict(ctx: dict[str, Any], merge_class_ok: bool) -> dict | None:
    if not (ctx["mergeable_state"] == "clean" and ctx["checks_state"] == "success" and merge_class_ok):
        return None
    cls = f"{ctx['author']} {ctx['level']} bump" if ctx["bot"] else ctx["allow"]
    return {
        "verdict": "safe_merge",
        "action": "merge",
        "reason": f"clean + checks green + allow-class ({cls})",
    }


def _pr_safe_close_verdict(item: dict, ctx: dict[str, Any], stale_days: int) -> dict | None:
    if item.get("superseded_by"):
        return {
            "verdict": "safe_close",
            "action": "close",
            "reason": f"superseded by {item['superseded_by']}",
        }
    age = item.get("age_days")
    if ctx["mergeable_state"] == "dirty" and isinstance(age, int) and age > stale_days:
        return {
            "verdict": "safe_close",
            "action": "close",
            "reason": f"stale {age}d + merge conflicts (abandoned) — confirm before close",
        }
    return None


def _pr_skip_reason(ctx: dict[str, Any], merge_class_ok: bool) -> str:
    ms, checks = ctx["mergeable_state"], ctx["checks_state"]
    why = []
    if ms and ms != "clean":
        why.append(f"mergeable_state={ms}")
    if checks and checks != "success":
        why.append(f"checks={checks}")
    if not merge_class_ok:
        why.append("not in auto-merge allow-class")
    return "; ".join(why) or "needs human review"


def _classify_pr(item: dict, stale_days: int, allow_major: bool, classes: set[str]) -> dict:
    if item.get("draft"):
        return {"verdict": "skip", "action": "none", "reason": "draft PR"}

    ctx = _pr_context(item)
    merge_class_ok = _pr_merge_class_ok(ctx, allow_major, classes)

    verdict = _pr_safe_merge_verdict(ctx, merge_class_ok)
    if verdict is not None:
        return verdict

    verdict = _pr_safe_close_verdict(item, ctx, stale_days)
    if verdict is not None:
        return verdict

    return {"verdict": "skip", "action": "none", "reason": _pr_skip_reason(ctx, merge_class_ok)}


def classify(item: dict, stale_days: int, allow_major: bool, classes: set[str]) -> dict:
    repo, num = item.get("repo", "?"), item.get("number", "?")
    verdict = (
        _classify_pr(item, stale_days, allow_major, classes)
        if _is_pr(item)
        else _classify_issue(item)
    )
    return {"repo": repo, "number": num, **verdict}


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="item JSON file(s); omit for stdin")
    ap.add_argument("--stale-days", type=int, default=60)
    ap.add_argument(
        "--allow-major",
        action="store_true",
        help="permit dependabot MAJOR bumps (off by default)",
    )
    ap.add_argument(
        "--allow-classes",
        default="dependabot-patch,dependabot-minor",
        help="comma list of auto-merge classes",
    )
    ap.add_argument("--format", choices=["json", "md"], default="md")
    return ap


def _load_blobs(files: list[str]) -> list[dict]:
    blobs: list[dict] = []
    if files:
        for f in files:
            with open(f) as fh:
                blobs.extend(_items(json.load(fh)))
    else:
        blobs.extend(_items(json.load(sys.stdin)))
    return blobs


def _print_markdown_report(verdicts: list[dict]) -> None:
    icon = {"safe_merge": "✅ merge", "safe_close": "🗑️ close", "skip": "⏭️ skip"}
    print("| Item | Verdict | Reason |")
    print("|------|---------|--------|")
    for v in verdicts:
        print(
            f"| {v['repo']}#{v['number']} | {icon.get(v['verdict'], v['verdict'])} | {v['reason']} |"
        )
    n_m = sum(v["verdict"] == "safe_merge" for v in verdicts)
    n_c = sum(v["verdict"] == "safe_close" for v in verdicts)
    n_s = sum(v["verdict"] == "skip" for v in verdicts)
    print(
        f"\n**{n_m} safe-merge · {n_c} safe-close · {n_s} skip** "
        f"(of {len(verdicts)}). Writes require explicit confirmation."
    )


def main() -> None:
    args = _build_arg_parser().parse_args()
    classes = {c.strip() for c in args.allow_classes.split(",") if c.strip()}

    blobs = _load_blobs(args.files)
    verdicts = [classify(i, args.stale_days, args.allow_major, classes) for i in blobs]
    if args.format == "json":
        print(json.dumps(verdicts, indent=2))
        return
    _print_markdown_report(verdicts)


if __name__ == "__main__":
    main()
