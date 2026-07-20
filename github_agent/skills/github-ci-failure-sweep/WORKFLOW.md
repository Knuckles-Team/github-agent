# Github Ci Failure Sweep

Sweeps GitHub Actions across one or more GitHub accounts (a user and/or an organization, all repositories) and reports which CI pipelines are currently failing, with the failing workflow, last failed run, and a probable cause. Use when asked which CI/pipelines are red, to audit Actions health across repos, to find failing builds for a user/org, or to triage broken CI before fixing it. Defaults to read-only analysis; re-running or cancelling runs is opt-in and runs only on explicit confirmation. Do NOT use for inspecting a single PR's checks (use github-tools) or for GitLab pipelines.

# GitHub CI Failure Sweep

## Overview

Cross-account GitHub Actions health check. Enumerates every repo under the target
accounts (e.g. the `example` user **and** the `Knuckles-Team` org), finds the
**latest failing run per workflow**, and emits a compact Markdown report grouped by
repo, with a probable cause per failure. Read-only by default; re-runs are opt-in.

Drives the **github-agent** MCP server. No `gh` CLI or local `GITHUB_*` token is
required — the MCP server holds its own auth.

## Tool access (works under delegation AND the multiplexer)

The tools are `github_actions`, `github_repos`, `github_orgs` — each takes
`action` + a `params_json` **JSON string**.

- **Under direct delegation** to `github-agent` (`execute_agent server=github-agent`,
  or the `mcp-client` skill) the tools are already bound by their **native** names
  (`github_actions`, `github_repos`, `github_orgs`) — call them directly. This is the
  path this skill is written for.
- **In the multiplexer / orchestrator** context the same tools carry the `gith__`
  prefix (`gith__actions`, …); mount them first with
  `load_tools(servers=["github-agent"])` (or `find_tools("github actions workflow runs")`),
  then call the prefixed names.

`github_actions` actions:
`list_workflows|list_runs|get_run|list_jobs|job_logs|trigger_dispatch|rerun|cancel|delete_run`.
See `references/actions-tool-cheatsheet.md`.

## Inputs
- **accounts**: `[{login, type}]`. Default:
  `[{login: "example", type: "user"}, {login: "Knuckles-Team", type: "org"}]`.
- **branch_scope**: `default` (only each repo's default branch — recommended) or `all`.

## Workflow

### Step 1 — Discover repositories
- User: `github_repos action=list`; keep repos owned by the login.
- Org: `github_orgs action=repos {"org": "<login>"}`.
- Record `default_branch`; skip archived/disabled repos.

### Step 2 — Fetch failing runs per repo (filtered)
For each repo, call `github_actions action=list_runs` filtered **server-side** to keep the
payload small:

```json
{"owner": "<login>", "repo": "<repo>", "status": "failure", "per_page": 15}
```

Add `"branch": "<default_branch>"` when `branch_scope=default`. The result is large and
will likely be spilled to a file by the harness — **do not read it raw**.

### Step 3 — Reduce to the latest failing pipeline per workflow
Feed each repo's result (file path or piped JSON) through the reducer, which keeps only the
latest run per `(repo, workflow, branch)` and only failing conclusions
(`failure|timed_out|cancelled|action_required|startup_failure`):

```bash
# one or many repo dumps at once
python scripts/summarize_runs.py repo1_runs.json repo2_runs.json --format json   # compact list
python scripts/summarize_runs.py repo1_runs.json --format md                      # report table
```

If every repo is green, the reducer reports "✅ No failing pipelines detected."

### Step 4 — Diagnose each failure (read the actual logs)
For each failing run in the reduced list, drill from run → failing job → log:
1. `github_actions action=list_jobs {"owner","repo","run_id"}` → find the job(s) whose
   `conclusion` is `failure`, and the failing step within.
2. `github_actions action=job_logs {"owner","repo","job_id"}` → the **decoded plaintext
   log** for that job. Pull ONE failing job at a time (logs are large; the harness may
   spill to a file — grep the tail for the error marker, don't read it whole).
3. Distill a one-line probable cause from the log evidence (e.g. "ruff lint failed",
   "pytest: 3 failures", "docker build timed out", "runner OOMKilled").

`github_actions action=get_run {"owner","repo","run_id"}` gives run-level metadata
(workflow, branch, head commit) if you need it. Cross-link the `github-backlog-planner`
skill if a failure is blocking an open PR.

### Step 5 — Present the report
Render the Markdown table (Step 3 `--format md`) augmented with the probable-cause notes
and a suggested fix per pipeline. Group by account → repo.

### Step 6 — Opt-in remediation (only on explicit confirmation)
Never trigger reruns without the user confirming. When asked:
- Re-run failed jobs: `github_actions action=rerun {"owner","repo","run_id"}`.
- Cancel a stuck run: `github_actions action=cancel {"owner","repo","run_id"}`.

## Related skills
- `github-backlog-planner` — full issue/PR backlog sweep + plan.
- `github-tools` — single-PR check inspection (do not duplicate here).
