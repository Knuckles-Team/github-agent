# `github_actions` cheatsheet (GitHub MCP)

`github_actions` takes `action` + a `params_json` string. Actions:
`list_workflows | list_runs | get_run | list_jobs | job_logs | trigger_dispatch | rerun | cancel | delete_run`.

> Under direct delegation the tool is `github_actions`; in the multiplexer it is
> `gith__actions` (mount with `load_tools(servers=["github-agent"])`). Same actions either way.

## Listing runs without blowing the token budget

`list_runs` returns full run objects (~13KB each). A naive call across many repos will
exceed the tool-output limit and spill to a file. Reduce the payload three ways:

1. **Filter server-side.** GitHub's runs endpoint accepts query params — pass them in
   `params_json`:
   - `"status": "failure"` — only failed runs (also: `cancelled`, `timed_out`, `action_required`).
   - `"branch": "<default_branch>"` — restrict to the branch CI gates on.
   - `"per_page": 10` — cap results.
   Example:
   ```json
   {"owner": "Knuckles-Team", "repo": "geniusbot", "status": "failure", "per_page": 10}
   ```
2. **Per-workflow latest.** Even filtered, you want the *latest* run per workflow, not the
   whole history. `scripts/summarize_runs.py` does this dedup.
3. **Never read the spilled file raw.** When the harness saves the result to a file, feed
   that file straight into the reducer:
   ```bash
   python scripts/summarize_runs.py /path/to/spilled_list_runs.json --format md
   ```

## Run object fields used

| Field | Use |
|---|---|
| `repository.full_name` | group by repo |
| `workflow_id`, `path`, `name` | identify the pipeline (`path` is cleanest; `name` may have a `#1234` suffix) |
| `head_branch` | branch the run was on |
| `status` | `completed` vs in-progress (only `completed` has a final result) |
| `conclusion` | `success` / `failure` / `timed_out` / `cancelled` / ... |
| `updated_at` | pick the latest run per workflow |
| `html_url`, `id`, `run_number`, `head_sha` | link + reference the run |

## Diagnosing a failure

Drill run → job → log, all via the MCP (no `gh` CLI needed):
1. `github_actions action=list_jobs {"owner","repo","run_id"}` → the jobs, each with a
   `conclusion`; find the `failure` one and its failing step.
2. `github_actions action=job_logs {"owner","repo","job_id"}` → the decoded plaintext log
   for that job. Pull ONE failing job at a time; if the harness spills it to a file, grep
   the tail for the error marker rather than reading it whole.
3. Write a one-line probable cause from the log evidence (e.g. "ruff lint failed",
   "pytest 3 failures", "docker build timeout", "runner OOMKilled").

`github_actions action=get_run {"owner","repo","run_id"}` returns run-level metadata if you
only need the workflow/branch/commit.

## Opt-in remediation (confirm first)
- Re-run failed jobs: `github_actions action=rerun {"owner","repo","run_id"}` (some servers
  expose a `failed_only` flag — re-running only failed jobs is preferred).
- Cancel a stuck run: `github_actions action=cancel {"owner","repo","run_id"}`.
