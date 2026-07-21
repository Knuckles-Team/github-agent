# Github Pr Review Sweep

Sweeps open pull requests across one or more GitHub accounts (a user and/or an organization, all repositories) and produces a per-PR review summary: for each open PR it gathers the diff size, mergeable state, required-review/check status, conflicts, and staleness, then classifies it as ready-to-merge, changes-requested, failing-checks, conflicts, or draft, with a recommended next action. Use when asked to review all open PRs across the org, audit pull-request health for a user/org, see which PRs are ready to merge or blocked, or triage review workload across repos. Defaults to read-only analysis; leaving a review, commenting, or merging is opt-in and runs only on explicit confirmation. Do NOT use for a single PR's checks (use github-tools), for issue/PR backlog planning (use github-backlog-planner), for CI failure triage (use github-ci-failure-sweep), or for GitLab merge requests.

# GitHub PR Review Sweep

## Overview

Cross-account open-pull-request review. Enumerates every repo under the target
accounts (e.g. the `example` user **and** the `Knuckles-Team` org), collects
every **open PR**, enriches each with mergeability + check + review signals, and
emits a Markdown report grouped by repo with a **verdict and recommended action per
PR**. Read-only by default; reviews/merges are opt-in.

Drives the **github-agent** MCP server. No `gh` CLI or local `GITHUB_*` token is
required — the MCP server holds its own auth.

## Tool access (works under delegation AND the multiplexer)

The tools are `github_pulls`, `github_actions`, `github_repos`, `github_orgs` —
each takes `action` + a `params_json` **JSON string**.

- **Under direct delegation** to `github-agent` (`execute_agent server=github-agent`,
  or the `mcp-client` skill) the tools are already bound by their **native** names
  — call them directly. This is the path this skill is written for.
- **In the multiplexer / orchestrator** context the same tools carry the `gith__`
  prefix (`gith__pulls`, …); mount them first with
  `load_tools(servers=["github-agent"])`.

`github_pulls` actions: `list|get|create|update|approve|request_reviewers|merge|
enable_auto_merge|disable_auto_merge`. See `references/pulls-tool-cheatsheet.md`
(includes the mergeable-state → verdict map).

## Inputs
- **accounts**: `[{login, type}]`. Default:
  `[{login: "example", type: "user"}, {login: "Knuckles-Team", type: "org"}]`.
- **include_drafts**: `true` (report drafts, flagged) or `false` (skip). Default `true`.

## Workflow

### Step 1 — Discover repositories
- User: `github_repos action=list`; keep repos whose `owner.login` == the login.
- Org: `github_orgs action=repos {"org": "<login>"}`.
- Record `default_branch`; skip archived/disabled repos.

### Step 2 — List open PRs per repo (filtered)
For each repo, `github_pulls action=list` filtered **server-side**:
```json
{"owner": "<login>", "repo": "<repo>", "state": "open", "per_page": 100, "max_pages": 0}
```
The result is large and will likely be spilled to a file by the harness — **do not read
it raw**; pass the file path(s) to the reducer.

### Step 3 — Reduce to one compact row per PR
Feed each repo's `list` dump through the reducer (pure JSON transform, stdlib only):
```bash
python scripts/summarize_prs.py repo1_pulls.json repo2_pulls.json --format md   # table
python scripts/summarize_prs.py *_pulls.json --format json                       # compact list
```
If no repo has an open PR, the reducer reports "✅ No open pull requests".
The reducer fills `repo, #, title, author, base←head, age_days, draft`; `mergeable`
and `size` stay `?`/`—` until enriched in Step 4.

### Step 4 — Enrich + diagnose each PR
For each open PR (skip drafts if `include_drafts=false`):
- `github_pulls action=get {"owner","repo","pull_number"}` → `mergeable_state`,
  `additions`, `deletions`, `review_comments`.
- Check status: `github_actions action=list_runs {"owner","repo","branch":"<head.ref>","per_page":10}`
  → latest run conclusion on the PR branch (match by `head_sha` for fork PRs).
- Re-run the reducer with `--detail <pr_get_*.json>` to merge `mergeable_state`/size into the table.
- Assign a **verdict** using the cheatsheet map: `clean`→✅ ready, `unstable`→⚠️ checks failing,
  `blocked`→⏳ needs review, `dirty`→❌ conflicts, `behind`→🔄 update branch, draft→📝 draft.
  Flag **stale** PRs (`age_days` over ~30) for follow-up.

### Step 5 — Present the report
Render the reducer's Markdown table augmented with the per-PR **verdict + recommended
action** (e.g. "approve & merge", "re-request review", "rebase onto base", "fix failing
checks — see github-ci-failure-sweep", "ping author — stale 45d"). Group by account → repo;
lead with a one-line tally (ready / blocked / conflicts / failing / draft).

### Step 6 — Opt-in actions (only on explicit confirmation)
Never write to GitHub without the user confirming. When asked:
- Comment / leave review feedback: `github_comments action=create {"owner","repo",
  "issue_number":N,"body":"<comment>"}` (a PR is an issue on GitHub). This is a
  read-only reporting/sweep skill by design — prefer handing gated writes off to
  `github-triage-resolver`, which re-verifies each item behind a dry-run +
  confirmation gate, rather than posting comments directly from here.
- Update branch / change base: `github_pulls action=update`.
- Merge: only when the user names the PR(s) and the verdict is ✅ ready.

## Related skills
- `github-ci-failure-sweep` — which pipelines are red (use for the failing-checks detail).
- `github-backlog-planner` — full open issue **and** PR backlog sweep + prioritized plan.
- `github-tools` — single-PR check inspection / review-comment fetch (do not duplicate here).
- `github-triage-resolver` — performs gated writes (comments, reviews) this skill only recommends.
