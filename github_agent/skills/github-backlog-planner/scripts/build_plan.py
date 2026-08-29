#!/usr/bin/env python3
"""Turn a collected list of GitHub issues/PRs into a prioritized Markdown plan.

Pure JSON -> Markdown transform. Performs NO network calls and needs NO auth: the
agent collects items via the ``github_*`` MCP tools (native names; under the
multiplexer these carry a ``gith__`` prefix) and feeds them here. This keeps
the script runnable in any environment (no ``gh`` CLI, no ``GITHUB_*`` token).

Input (stdin or --input PATH): a JSON list of item objects. Each item:
    {
      "account":   "Knucklessg1" | "Knuckles-Team",   # owner / org login
      "repo":      "geniusbot",                          # repo name (no owner)
      "kind":      "issue" | "pr",
      "number":    42,
      "title":     "...",
      "url":       "https://github.com/...",             # html_url
      "state":     "open",
      "status":    "addressed" | "in-progress" | "needs-action",  # triage outcome
      "evidence":  "merged PR #51 landed fix in foo.py",  # why this status (free text)
      "recommendation": "Close as fixed by #51",          # concrete next step
      "priority":  "high" | "medium" | "low",            # optional, default medium
      "labels":    ["bug", ...]                            # optional
    }

Output: a grouped, prioritized Markdown remediation plan on stdout.

Usage:
    python build_plan.py < items.json
    python build_plan.py --input items.json --title "Backlog Plan"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

STATUS_ORDER = {"needs-action": 0, "in-progress": 1, "addressed": 2}
STATUS_LABEL = {
    "needs-action": "🔴 Needs action",
    "in-progress": "🟡 In progress",
    "addressed": "🟢 Addressed",
}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
PRIORITY_BADGE = {"high": "P1", "medium": "P2", "low": "P3"}


def _or_default(value: Any, default: Any) -> Any:
    """Return ``value`` unless it is falsy, in which case return ``default``."""
    return value if value else default


def _normalize_choice(value: Any, valid: dict[str, int], default: str) -> str:
    """Lowercase/strip ``value`` and fall back to ``default`` if not in ``valid``."""
    choice = str(_or_default(value, default)).strip().lower()
    return choice if choice in valid else default


def _norm(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "account": _or_default(item.get("account"), "(unknown)"),
        "repo": _or_default(item.get("repo"), "(unknown)"),
        "kind": str(_or_default(item.get("kind"), "issue")).strip().lower(),
        "number": item.get("number"),
        "title": str(_or_default(item.get("title"), "(no title)")).strip(),
        "url": _or_default(item.get("url"), ""),
        "status": _normalize_choice(item.get("status"), STATUS_ORDER, "needs-action"),
        "evidence": str(_or_default(item.get("evidence"), "")).strip(),
        "recommendation": str(_or_default(item.get("recommendation"), "")).strip(),
        "priority": _normalize_choice(item.get("priority"), PRIORITY_ORDER, "medium"),
        "labels": _or_default(item.get("labels"), []),
    }


def _sort_key(it: dict[str, Any]) -> tuple:
    return (
        STATUS_ORDER[it["status"]],
        PRIORITY_ORDER[it["priority"]],
        0 if it["kind"] == "pr" else 1,
        it["number"] if isinstance(it["number"], int) else 1 << 30,
    )


def _summary_counts(
    items: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    by_status: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    for it in items:
        by_status[it["status"]] += 1
        by_kind[it["kind"]] += 1
    return by_status, by_kind


def _summary_line(items: list[dict[str, Any]]) -> str:
    by_status, by_kind = _summary_counts(items)
    return (
        f"**{len(items)} open item(s)** — "
        f"{by_kind.get('issue', 0)} issue(s), {by_kind.get('pr', 0)} PR(s). "
        f"{by_status.get('needs-action', 0)} need action, "
        f"{by_status.get('in-progress', 0)} in progress, "
        f"{by_status.get('addressed', 0)} likely closable."
    )


def _closable_lines(items: list[dict[str, Any]]) -> list[str]:
    """Render the 'safe to close' section for items marked addressed."""
    closable = sorted((i for i in items if i["status"] == "addressed"), key=_sort_key)
    if not closable:
        return []
    lines = ["## ✅ Verified addressed — safe to close", ""]
    for it in closable:
        ref = f"{it['account']}/{it['repo']}#{it['number']}"
        lines.append(f"- [{ref}]({it['url']}) — {it['title']}")
        if it["evidence"]:
            lines.append(f"  - _Why:_ {it['evidence']}")
        if it["recommendation"]:
            lines.append(f"  - _Action:_ {it['recommendation']}")
    lines.append("")
    return lines


def _group_outstanding(items: list[dict[str, Any]]) -> dict[str, dict[str, list]]:
    """Group non-addressed items by account -> repo."""
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        if it["status"] == "addressed":
            continue  # covered by _closable_lines instead
        grouped[it["account"]][it["repo"]].append(it)
    return grouped


def _repo_table_row(it: dict[str, Any]) -> str:
    badge = PRIORITY_BADGE[it["priority"]]
    num = f"[#{it['number']}]({it['url']})" if it["url"] else f"#{it['number']}"
    kind = "PR" if it["kind"] == "pr" else "issue"
    title = it["title"].replace("|", "\\|")
    rec = (it["recommendation"] or "—").replace("|", "\\|")
    status = STATUS_LABEL[it["status"]]
    return f"| {badge} | {num} | {kind} | {title} | {status} | {rec} |"


def _repo_table_lines(repo: str, repo_items: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(repo_items, key=_sort_key)
    lines = [
        f"#### `{repo}` ({len(ordered)})",
        "",
        "| | # | Type | Title | Status | Recommended next step |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(_repo_table_row(it) for it in ordered)
    lines.append("")
    return lines


def _action_plan_lines(items: list[dict[str, Any]]) -> list[str]:
    lines = ["## Action plan", ""]
    grouped = _group_outstanding(items)
    if not grouped:
        lines.append("_No outstanding items requiring action._")
        lines.append("")
        return lines
    for account in sorted(grouped):
        lines.append(f"### {account}")
        lines.append("")
        for repo in sorted(grouped[account]):
            lines.extend(_repo_table_lines(repo, grouped[account][repo]))
    return lines


def build_markdown(items: list[dict[str, Any]], title: str) -> str:
    items = [_norm(i) for i in items]
    out: list[str] = [f"# {title}", "", _summary_line(items), ""]
    out.extend(_closable_lines(items))
    out.extend(_action_plan_lines(items))
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="Path to items JSON (default: stdin).")
    ap.add_argument(
        "--title",
        default="GitHub Backlog Remediation Plan",
        help="Plan title.",
    )
    args = ap.parse_args(argv)

    raw = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    if not raw.strip():
        print("error: no input JSON provided", file=sys.stderr)
        return 2
    data = json.loads(raw)
    # Accept either a bare list or {"items": [...]}.
    items = data["items"] if isinstance(data, dict) else data
    if not isinstance(items, list):
        print("error: expected a JSON list of items", file=sys.stderr)
        return 2

    sys.stdout.write(build_markdown(items, args.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
