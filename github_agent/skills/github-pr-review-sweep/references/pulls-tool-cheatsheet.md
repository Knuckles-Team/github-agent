# `github_pulls` / `github_actions` cheatsheet (github-agent)

Read just-in-time when driving the sweep. Tool names below are the **native** names
(the GitHub MCP server, `github-agent`, holds its own auth — no `gh` CLI / `GITHUB_*`
token needed). Under direct delegation to `github-agent` these are already bound and
callable directly; under the multiplexer/orchestrator the same tools carry the
`gith__` prefix (`gith__pulls`, `gith__actions`) and must be mounted first with
`load_tools(servers=["github-agent"])`.

## `github_pulls` — `action` + `params_json`
| action | params_json | returns |
|--------|-------------|---------|
| `list` | `{"owner","repo","state":"open","per_page":100,"max_pages":0}` | open PRs for a repo (basic objects: number, title, user, draft, base/head, created_at, comments) |
| `get`  | `{"owner","repo","pull_number"}` | full PR: `mergeable`, `mergeable_state` (`clean`/`dirty`/`blocked`/`behind`/`unstable`), `additions`, `deletions`, `changed_files`, `review_comments` |
| `update` | `{"owner","repo","pull_number", ...}` | mutate (title/body/base/state) — opt-in only |
| `create` | `{"owner","repo","title","head","base", ...}` | open a PR — not used by this sweep |
| `approve` / `request_reviewers` | `{"owner","repo","pull_number", ...}` | review actions — not used by this sweep (read-only) |
| `merge` / `enable_auto_merge` / `disable_auto_merge` | `{"owner","repo","pull_number", ...}` | merge actions — opt-in only |

Notes:
- `list` does NOT include `mergeable_state` or check status — fetch those with `get` (and `github_actions`) per PR in the diagnose step.
- `max_pages: 0` pages through everything; results are large and the harness spills them to a file — feed that file to `scripts/summarize_prs.py`, do not read it raw.
- There is no `comment` action on `github_pulls`/`github_issues`, and no dedicated comment tool — this sweep is read-only reporting; leaving comments is out of scope (hand off to `github-triage-resolver`).

## Checks for a PR's head
- `github_actions action=list_runs {"owner","repo","branch":"<pr.head.ref>","per_page":10}` → latest runs on the PR branch; conclusion `success`/`failure`/… is the checks signal.
- (`head.ref` comes from the PR object; for forks the branch lives under the fork — fall back to the run whose `head_sha` matches the PR head SHA.)

## Repo enumeration
- User account: `github_repos action=list` → keep repos whose `owner.login` == the user.
- Org account: `github_orgs action=repos {"org":"<login>"}` (or `github_repos` org variant).
- Record `default_branch`; skip `archived`/`disabled` repos.

## Mergeable-state → verdict mapping
| `mergeable_state` | meaning | verdict |
|-------------------|---------|---------|
| `clean` | mergeable, checks green, approved/!required | ✅ ready to merge |
| `unstable` | mergeable but a non-required check is failing/pending | ⚠️ checks failing/pending |
| `blocked` | required review or status not satisfied | ⏳ needs review / required check |
| `dirty` | merge conflict | ❌ conflicts — rebase needed |
| `behind` | base moved; needs update | 🔄 update branch |
| `draft` (PR.draft=true) | work in progress | 📝 draft |
