#!/usr/bin/env python3
"""Reduce verbose GitHub Actions ``list_runs`` output to the failing pipelines.

``github_actions action=list_runs`` returns large payloads (~65KB/repo) that the
harness spills to a file. Never read that raw — feed it here. This script:
  * accepts one or more ``list_runs`` result files (or stdin),
  * keeps only the **latest** run per ``(repo, workflow, branch)``,
  * filters to runs whose conclusion is a failure
    (``failure|timed_out|cancelled|action_required|startup_failure``),
  * emits a compact JSON list and/or a Markdown table.

It is a pure JSON transform: no network, no auth, stdlib only. Works on either a
raw MCP result object ``{"status","message","data":[...]}`` or a bare list of run
objects, and tolerates the GitHub-native ``{"workflow_runs":[...]}`` shape too.

Usage:
    python summarize_runs.py repo1_runs.json repo2_runs.json --format md
    cat runs.json | python summarize_runs.py --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

FAILURE_CONCLUSIONS = {
    "failure",
    "timed_out",
    "cancelled",
    "action_required",
    "startup_failure",
}


def _dict_items(value: Any) -> list[dict] | None:
    """``value`` filtered to its dict elements, if it is a list; else ``None``."""
    if not isinstance(value, list):
        return None
    return [r for r in value if isinstance(r, dict)]


def _runs_under_key(blob: dict, key: str) -> list[dict] | None:
    """Runs found at ``blob[key]``, directly or one level deeper under
    ``workflow_runs`` (the GitHub-native shape), or ``None`` if absent."""
    val = blob.get(key)
    direct = _dict_items(val)
    if direct is not None:
        return direct
    if isinstance(val, dict):
        return _dict_items(val.get("workflow_runs"))
    return None


def _extract_runs(blob: Any) -> list[dict]:
    """Pull the list of run objects out of whatever shape we were handed."""
    direct = _dict_items(blob)
    if direct is not None:
        return direct
    if not isinstance(blob, dict):
        return []
    for key in ("data", "workflow_runs"):
        found = _runs_under_key(blob, key)
        if found is not None:
            return found
    return []


def _repo_of(run: dict) -> str:
    repo = run.get("repository")
    if isinstance(repo, dict):
        return repo.get("full_name") or repo.get("name") or "(unknown)"
    return run.get("repo") or "(unknown)"


def _workflow_label(run: dict) -> str:
    # `name` can carry a noisy "#1234" suffix for dynamic runs; prefer `path`.
    path = run.get("path")
    if path:
        return str(path)
    name = run.get("name") or ""
    return name.split(" #", 1)[0].strip() or f"workflow {run.get('workflow_id')}"


def _is_completed(run: dict) -> bool:
    return (run.get("status") or "").lower() == "completed"


def _run_group_key(run: dict) -> tuple:
    return (
        _repo_of(run),
        run.get("workflow_id") or _workflow_label(run),
        run.get("head_branch"),
    )


def _latest_by_group(runs: list[dict]) -> dict[tuple, dict]:
    """Most-recently-updated completed run per (repo, workflow, branch)."""
    latest: dict[tuple, dict] = {}
    for run in runs:
        if not _is_completed(run):
            continue  # in-progress/queued runs have no final conclusion yet
        key = _run_group_key(run)
        prev = latest.get(key)
        if prev is None or str(run.get("updated_at") or "") > str(
            prev.get("updated_at") or ""
        ):
            latest[key] = run
    return latest


def _is_failing(run: dict) -> bool:
    return (run.get("conclusion") or "").lower() in FAILURE_CONCLUSIONS


def _failing_record(run: dict) -> dict:
    return {
        "repo": _repo_of(run),
        "workflow": _workflow_label(run),
        "branch": run.get("head_branch"),
        "conclusion": run.get("conclusion"),
        "run_id": run.get("id"),
        "run_number": run.get("run_number"),
        "event": run.get("event"),
        "updated_at": run.get("updated_at"),
        "html_url": run.get("html_url"),
        "head_sha": (run.get("head_sha") or "")[:8],
    }


def reduce_runs(runs: list[dict]) -> list[dict]:
    """Latest run per (repo, workflow, branch); keep only failing ones."""
    latest = _latest_by_group(runs)
    failing = [_failing_record(run) for run in latest.values() if _is_failing(run)]
    failing.sort(key=lambda r: (r["repo"], r["workflow"], r["branch"] or ""))
    return failing


def to_markdown(failing: list[dict]) -> str:
    if not failing:
        return "## CI failure sweep\n\n✅ No failing pipelines detected.\n"
    out = ["## CI failure sweep", "", f"**{len(failing)} failing pipeline(s):**", ""]
    by_repo: dict[str, list[dict]] = {}
    for f in failing:
        by_repo.setdefault(f["repo"], []).append(f)
    for repo in sorted(by_repo):
        out.append(f"### `{repo}`")
        out.append("")
        out.append("| Workflow | Branch | Result | Last failure | Run |")
        out.append("|---|---|---|---|---|")
        for f in by_repo[repo]:
            run = f"[{f['run_id']}]({f['html_url']})" if f["html_url"] else f["run_id"]
            out.append(
                f"| {f['workflow']} | {f['branch'] or '—'} | {f['conclusion']} "
                f"| {f['updated_at'] or '—'} | {run} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="list_runs JSON files (default: stdin).")
    ap.add_argument(
        "--format", choices=["md", "json"], default="md", help="Output format."
    )
    args = ap.parse_args(argv)

    runs: list[dict] = []
    sources = (
        [open(p, encoding="utf-8").read() for p in args.files]
        if args.files
        else [sys.stdin.read()]
    )
    for raw in sources:
        if not raw.strip():
            continue
        runs.extend(_extract_runs(json.loads(raw)))

    failing = reduce_runs(runs)
    if args.format == "json":
        json.dump(failing, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(to_markdown(failing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
