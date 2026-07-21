---
name: github-backlog-planner
skill_type: skill
description: >-
  Sweeps every open issue and pull request across one or more GitHub accounts
  (a user and/or an organization, all repositories), deeply verifies which items
  are already addressed, and emits a prioritized Markdown remediation plan. Use
  when asked to "go through all issues and PRs", triage a whole backlog, plan
  what to work on next across repos, or audit open work for a user/org. Defaults
  to read-only analysis; write actions (close, comment, open tracking issue) are
  opt-in and run only on explicit confirmation. Do NOT use for a single PR's
  review comments (use github-tools) or interactive one-tracker triage (use
  issue-triage).
license: MIT
tags: [github, issues, pull-requests, backlog, triage, planning, ops]
metadata:
  author: Genius
  version: '1.0.4'
---

# GitHub Backlog Planner

## Overview

Account-wide backlog sweep. Discovers all open issues and PRs across the target
accounts (e.g. the `Knucklessg1` user **and** the `Knuckles-Team` org), determines
which are **already addressed** (deep verification, not just labels), and produces a
single prioritized Markdown plan grouped by account → repo. Read-only by default;
any change to GitHub is opt-in.

Drives the **github-agent** MCP server. No `gh` CLI or local `GITHUB_*` token is
required — the MCP server holds its own auth.

## Tool access (works under delegation AND the multiplexer)

The tools are `github_issues`, `github_pulls`, `github_search`, `github_repos`,
`github_orgs`, `github_commits`, `github_contents` — each takes `action` + a
`params_json` **JSON string**.

- **Under direct delegation** to `github-agent` (`execute_agent server=github-agent`,
  or the `mcp-client` skill) the tools are already bound by their **native** names
  — call them directly. This is the path this skill is written for.
- **In the multiplexer / orchestrator** context the same tools carry the `gith__`
  prefix (`gith__issues`, …); mount them first with
  `load_tools(servers=["github-agent"])` (or
  `find_tools("github issues pull requests repos")`).

Confirmed action shapes: `github_issues`/`github_pulls` → `list|get|create|update`;
`github_search` → `repositories|issues|code`; `github_repos` → `list|get`;
`github_orgs` → `repos|get`.

## Inputs

- **accounts**: list of `{login, type}` where type is `user` or `org`. Default:
  `[{login: "Knucklessg1", type: "user"}, {login: "Knuckles-Team", type: "org"}]`.
- **include_prs**: default true. **include_issues**: default true.

## Workflow

### Step 1 — Discover repositories
- User: `github_repos action=list` (authenticated user) — keep repos owned by the target login.
- Org: `github_orgs action=repos {"org": "<login>"}`.
- Record each repo's `default_branch`, `open_issues_count`, and archived flag. Skip archived repos.

### Step 2 — Bulk-discover open items (efficient path)
Use search, which returns issues AND PRs compactly account-wide:
- `github_search action=issues {"q": "user:Knucklessg1 is:open is:issue"}`
- `github_search action=issues {"q": "user:Knucklessg1 is:open is:pr"}`
- `github_search action=issues {"q": "org:Knuckles-Team is:open is:issue"}`
- `github_search action=issues {"q": "org:Knuckles-Team is:open is:pr"}`

Paginate (`per_page`, `page`) until exhausted. If search results look incomplete for a
repo, fall back to per-repo `github_issues action=list {owner, repo, state:"open"}` and
`github_pulls action=list {owner, repo, state:"open"}`. Remember: the GitHub search/issues
APIs return PRs in the issues list — distinguish them by the `pull_request` field.

### Step 3 — Heuristic triage (Pass 1)
Bucket every item using `references/addressed-heuristics.md` (Pass 1 table) into
`needs-action`, `in-progress`, or *maybe addressed*. This is cheap and uses only the
data already fetched.

### Step 4 — Deep verification (Pass 2)
For each *maybe addressed* item, follow Pass 2 in `references/addressed-heuristics.md`:
`github_issues/pulls action=get` for linked PRs/commits, then `github_commits action=get`
and/or `github_contents action=get` to **confirm the fix actually landed on the default
branch** before marking it `addressed`. For items that still need a plan, read the
relevant source to write a one-line root-cause note and a concrete recommendation.

### Step 5 — Build the plan
Normalize each item to the JSON schema in `scripts/build_plan.py` (account, repo, kind,
number, title, url, state, status, evidence, recommendation, priority, labels), then:

```bash
python scripts/build_plan.py --input items.json --title "GitHub Backlog Plan (<date>)"
# or: cat items.json | python scripts/build_plan.py
```

This renders: a summary line, a "verified addressed — safe to close" section, and a
per-account/per-repo action table sorted by status then priority. Present it to the user.

### Step 6 — Opt-in actions (only on explicit confirmation)
Never write to GitHub without the user confirming. When asked:
- Close a verified-addressed issue: `github_issues action=update {owner, repo, number, state:"closed"}`.
- Post a triage comment: `github_comments action=create {owner, repo, issue_number, body}`
  (prefix the body with `> *This was generated by AI during triage.*`, mirroring the
  `issue-triage` convention). Prefer handing gated/bulk writes off to
  `github-triage-resolver` instead of calling this directly — it re-verifies each item
  and runs behind the same dry-run + confirmation gate this skill only recommends.
- Open a tracking issue for orphaned work: `github_issues action=create`.

## Related skills
- `github-ci-failure-sweep` — cross-link when a PR is blocked by failing CI.
- `github-tools` — single-PR comments/CI inspection (do not duplicate here).
- `issue-triage` — interactive, single-tracker state-machine triage.
- `github-triage-resolver` — performs gated writes (comments, closes) this skill only recommends.
