# Github Org Remediation Loop

Sweeps org-wide GitHub issues, failing CI runs, and open incoming pull requests, validates each item, generates an SDD spec, risk-classifies it, and either auto-implements + auto-merges + closes (low risk) or implements on a branch and opens a pending-review PR (elevated risk). mode=issues_ci runs the issue/CI-failure remediation loop (Loop A); mode=incoming_prs runs the incoming-PR triage-or- supersede loop (Loop B). Drives github-agent native tools (github_issues, github_actions, github_pulls, github_branches, github_contents) plus the spec-generator / sdd-implementer / spec-verifier universal-skills atomic skills. Use when asked to "clean up the backlog and actually fix things", "turn failing CI into tracked fixes", "process incoming PRs", "remediate org CI failures", "auto-fix issues", or "triage and merge org PRs". Read-only sweep + spec + risk-classify always happen first; every write (branch/commit/PR/merge/close) is gated by autonomy level, defaulting to dry-run. Delegation-ready: runnable unattended via graph_orchestrate execute_agent against the github-agent. Do NOT use for a one-off single-PR review (github-pull-request-review) or for a report-only sweep with no fixing (github-backlog-planner / github-pr-review-sweep alone) — this skill is composition-first and introduces no new business logic beyond the fix-risk gate in scripts/classify_fix_risk.py.

# GitHub Org Remediation Loop

The **orchestrating** skill over the GitHub sweep/triage fleet: it doesn't just
report or triage, it turns a swept item into an SDD spec, risk-gates the fix, and
either lands it autonomously or hands a human a pending-review PR. It is a thin
DAG over existing atomic skills/tools — it adds exactly one new piece of logic
(`scripts/classify_fix_risk.py`, the fix-risk gate); everything else composes
`github-ci-failure-sweep`, `github-issue-tracking`, `github-backlog-planner`,
`github-pr-review-sweep`, `github-triage-resolver`, and the universal-skills SDD
atomics (`spec-generator`, `spec-intake-wizard`, `sdd-implementer`,
`spec-verifier`).

Full design rationale, inventory, and gap analysis:
`reports/spec-driven-remediation-loop-design.md`.

## Tool access (works under delegation AND the multiplexer)

Native github-agent tools used: `github_issues`, `github_actions`, `github_pulls`,
`github_branches`, `github_contents`. Each takes `action` + a `params_json` JSON
string.

- **Under direct delegation** to `github-agent` (`execute_agent server=github-agent`,
  `graph_orchestrate action=execute_agent`, or the `mcp-client` skill) the tools are
  bound by their **native** names — call them directly. This is the path this skill
  is written for.
- **In the multiplexer / orchestrator** context the same tools carry the `gith__`
  prefix (`gith__issues`, `gith__actions`, `gith__pulls`, `gith__branches`,
  `gith__contents`) — mount them first with `load_tools(servers=["github-agent"])`
  (or `find_tools("github issues actions pulls branches contents")`), then call the
  prefixed names. The `SKILL.md`s of the sibling skills named above document each
  tool's `action=` surface in full; this skill does not re-document it.

`github_pulls` actions relevant here: `create`, `enable_auto_merge`,
`disable_auto_merge`, `merge`, `update` (all guarded — `merge`/`enable_auto_merge`
require `allow_destructive=true`). `github_issues` actions: `list`, `get`, `create`,
`update` (no native comment action — see Step 6 for the AI-disclaimer-comment
convention). `github_branches`: `create` (from the repo's default branch SHA).
`github_actions`: `list_runs`, `list_jobs`, `job_logs`, `rerun`.

## Inputs

- **mode**: `issues_ci` (Loop A) | `incoming_prs` (Loop B). Required.
- **accounts**: `[{login, type}]`, default
  `[{login:"example",type:"user"},{login:"Knuckles-Team",type:"org"}]` (matches
  the sibling sweep skills).
- **repos**: optional list of `owner/repo` — overrides `accounts` to scope a run to
  specific repositories (use this for a first test run; an org-wide sweep on the
  first run can open far more branches/PRs than intended).
- **autonomy**: `dry-run` (default — plan only, no writes) | `confirm-each` (act
  per item on explicit yes) | `auto-safe` (act without per-item prompting, but
  ONLY on `low_risk`/`safe_merge` verdicts — never on `elevated_risk`/`skip`).
  Identical semantics to `github-triage-resolver`.
- **max_items**: cap on items processed per run (avoid an org-wide sweep turning
  into dozens of simultaneous branches on the first run). Default `5`.
- **allow_major_bump**: bool, default `false` — forwarded to `classify_safe.py
  --allow-major` for Loop B's existing-item leg.

## Risk model (two complementary classifiers)

This loop uses **two** deterministic, stdlib-only classifiers, run at different
points in the DAG — never mix them up:

1. **`github-triage-resolver`'s `scripts/classify_safe.py`** — the *existing-item*
   leg. Answers "is this **already-open** PR/issue mergeable/closable as-is."
   Verdicts: `safe_merge` / `safe_close` / `skip`. Used for Loop B's incoming PRs
   (step 4) and Loop A's "is this issue already fixed" pre-check (step 2).
2. **This skill's own `scripts/classify_fix_risk.py`** — the *diff-risk* leg.
   Answers "is the fix **we just wrote** safe to auto-merge." It only runs once
   `sdd-implementer` has produced a diff (Loop A always; Loop B only when step 5
   re-implements). Verdicts: `low_risk` / `elevated_risk`.

### `classify_fix_risk.py` rule (the one new piece of logic)

```
low_risk       IF   spec-verifier CHECKLIST.md is 100% pass
               AND  diff touches <= 5 files (--max-files) and <= 150 lines (--max-lines)
               AND  diff touches NONE of: migrations/, schema/, auth/, secrets/,
                    .github/workflows/, bumpversion config, Dockerfile/compose
                    (infra surfaces — override via --infra-patterns)
               AND  all required CI checks are green on the fix branch
               AND  the fix is evidence-confirmed (issue: a regression test
                    reproduces the original report and now passes; CI: the
                    specific failing job passes on rerun)
elevated_risk  OTHERWISE — the reason lists every gate that failed
```

```bash
python scripts/classify_fix_risk.py items.json --format md
python scripts/classify_fix_risk.py items.json --format json --max-files 5 --max-lines 150
```

Both classifiers are conservative by design: when in doubt, the diff routes to
`elevated_risk` and the existing item routes to `skip`.

## Workflow

### Step 1 — Intake (sweep)
- **mode=issues_ci**: fan out two parallel sweeps —
  `github-ci-failure-sweep` (`github_actions action=list_runs status=failure` →
  `list_jobs` → `job_logs` for the probable cause) **‖**
  `github-issue-tracking` / `github-backlog-planner`
  (`github_issues action=list org=<org>` org-wide, or per-repo if `repos` is set).
- **mode=incoming_prs**: `github-pr-review-sweep`
  (`github_pulls action=list state=open` + `github_actions action=list_runs` for
  check status).

Cap the combined item list at `max_items` (prioritize CI failures, then oldest
issues/PRs).

### Step 2 — Validate [depends_on: Step 1]
Drop anything already-addressed/flaky/abandoned before spending an SDD cycle on it:
- **Issues**: `github-backlog-planner`'s Pass-2 deep-verification heuristics
  (`references/addressed-heuristics.md` there) — confirm via
  `github_issues action=get` + a linked commit/PR that it is NOT already fixed on
  the default branch.
- **CI failures**: rerun the red job ONCE — `github_actions action=rerun` — to
  rule out flake before opening a spec for it. If it goes green, log it as
  skipped (flaky), not a remediation item.
- **Incoming PRs**: `github-triage-resolver`'s evidence-gathering (`github_pulls
  action=get` for `mergeable_state`, `github_actions action=list_runs` on the PR
  head branch) plus `scripts/classify_safe.py` — this doubles as Loop B's first
  risk pass (see Step 4).

Log every dropped item with its reason; it does not get a spec.

### Step 3 — Spec [depends_on: Step 2] — `spec-generator`
One `spec.md` + `Spec` JSON per validated item
(`agent_data/specs/{feature_id}.md`, per the Dual-Write convention), seeded from
the issue body / CI log excerpt / PR title+body+diff summary. Use
`spec-intake-wizard` first only if the item's intent is genuinely ambiguous (rare
for a CI-log or a well-formed issue).

### Step 4 — Risk-classify (existing-item leg) [depends_on: Step 3]
- **mode=incoming_prs**: `scripts/classify_safe.py` on the enriched PR from Step
  2. `safe_merge` → skip straight to Step 6a (no re-implementation — merge the
  incoming PR as-is). `safe_close` / `skip` → the PR needs rework: continue to
  Step 5 (re-implement).
- **mode=issues_ci**: this leg is Step 2's "already fixed" check; nothing further
  to classify here — always continue to Step 5.

### Step 5 — Implement [depends_on: Step 3] — `sdd-implementer`, then `spec-verifier`
Skipped only for `mode=incoming_prs` items already resolved `safe_merge` in Step 4.
Otherwise:
1. `github_branches action=create` off the repo's default branch
   (`<feature_id>-fix` naming).
2. `sdd-implementer` executes `tasks.md` against the branch — file edits, tests.
3. `github_contents action=update` / `action=create` to commit the changes (or the
   implementer's own git-native commit path if running with local repo access).
4. `spec-verifier` produces `CHECKLIST.md` / `DRIFT_REPORT.md` under
   `.specify/specs/{feature_id}/`.

### Step 6 — Risk-classify (diff-risk leg) [depends_on: Step 5]
Run `scripts/classify_fix_risk.py` on the produced diff (files/lines changed,
touched paths, `spec-verifier` pass/fail, CI checks on the fix branch, evidence
confirmation). `low_risk` → Step 7a. `elevated_risk` → Step 7b.

### Step 7a — LOW RISK: auto-merge + close [depends_on: Step 6]
Only proceeds under `autonomy=auto-safe` or an explicit per-item confirmation.
1. `github_pulls action=create` (fix branch → default branch).
2. `github_pulls action=enable_auto_merge allow_destructive=true`.
3. Once required checks report green, `github_pulls action=merge
   allow_destructive=true`.
4. **mode=issues_ci** (issue source): `github_issues action=update
   state=closed` on the source issue, with a comment (see below) — the AI
   disclaimer, a link to the merged PR, and the spec/checklist path. (CI-failure
   source: no issue to close — the merged fix + green rerun IS the closure.)
   **mode=incoming_prs** (already `safe_merge` from Step 4): `github_pulls
   action=merge allow_destructive=true` directly — no re-implementation, no PR of
   our own.
5. **Comment convention**: `github_issues`/`github_pulls` have no native comment
   action. Reuse `github-triage-resolver`'s
   `../github-triage-resolver/scripts/gh_write.py comment <owner/repo> <number>
   --body "<body>" --confirm`. Every comment body MUST start with
   `> *This was generated by AI during triage.*` per that skill's convention —
   do not invent a second disclaimer wording.

### Step 7b — ELEVATED RISK: PR pending human review [depends_on: Step 6]
1. `github_pulls action=create` **without** `enable_auto_merge`.
2. `github_pulls action=update` to add the `needs-human-review` label.
3. Comment (via `gh_write.py`, same disclaimer convention) cross-linking the
   source issue/CI run/incoming PR and the `spec.md`/`DRIFT_REPORT.md` paths.
4. **mode=incoming_prs supersession** (Step 4 was NOT `safe_merge`, so Step 5
   re-implemented): once **our** replacement PR from step 1 above is *created*
   (not merged), close the **original incoming PR**:
   `github_pulls action=update state=closed` with a comment pointing to the
   replacement PR. Never leave two open PRs for the same intent. The human then
   merges our PR via the normal `github_pulls action=merge` (or
   `github-pr-review-sweep`).
5. Step 8 (dev-cycle close-out) is deferred to the human's merge — do not run it here.

### Step 8 — Dev-cycle close-out [depends_on: Step 7a] (Step 7b defers this)
After a low-risk fix merges to the repo's default branch:
1. `bump-my-version bump patch` (or `minor`/`major` per the fix's actual scope —
   almost always `patch` for a remediation fix) — never hand-edit a version
   string (see this package's `AGENTS.md` version-drift edict).
2. Merge the fix branch to `main` if not already fast-forwarded by the PR merge
   itself (`rm_worktree merge <repo> <branch> --into main` via the
   `repository-manager` MCP, or `git merge --no-ff`).
3. Prune the worktree: `rm_worktree remove <repo> <branch> --delete-branch`;
   `rm_worktree prune`.
4. Push to **both** remotes per the repo's dual-remote convention — GitHub (this
   skill's own writes already went there) **and** GitLab, if the repo mirrors —
   `git push github main --tags && git push gitlab main --tags`.

### Step 9 — KG persistence [depends_on: Step 8]
`graph_write` / `kg_ingest` on `.specify/` + the changed files (per
`sdd-implementer`'s Dual-Write principle). For the CI leg, `github_ingest_pipelines`
records the `:PipelineRun`/`:CheckRun` nodes. Record the risk verdict (both
classifiers) and the autonomy level used as run metadata, then
`graph_feedback(action_outcome)` so future runs can tell `auto-safe` items that
were correctly gated from ones that should have routed to `elevated_risk`.

## Execution / delegation

Same dual-mode contract every universal-skills workflow uses: if graph-os is
reachable, offload the whole DAG via `graph_orchestrate action=execute_workflow`
(or the `kg-delegate` skill) for parallel/swarm execution; otherwise execute the
steps natively in `depends_on` order — steps with no unmet dependency run in
parallel, then their dependents.

Concretely, delegation-ready via:

```
graph_orchestrate(action="execute_agent", agent_name="github-agent",
  hints_json={"skill": "github-org-remediation-loop",
              "params": {"mode": "issues_ci",
                         "repos": ["Knuckles-Team/github-agent"],
                         "autonomy": "dry-run", "max_items": 1}})
```

or, once ingested, `action="execute_workflow"` against the ingested workflow
definition directly.

## Safety (non-negotiable)

- Dry-run by default; every write (branch/commit/PR/merge/close) requires
  `confirm-each` confirmation or an explicit `auto-safe` opt-in.
- `auto-safe` acts ONLY on `low_risk` (Step 7a) / `safe_merge` (Loop B's direct
  merge) — it never touches an `elevated_risk`/`skip` item without a human.
- Never merge with failing/pending required checks or a non-`clean` mergeable
  state; never force-merge; never override branch protection; never auto-merge a
  diff that touches an infra surface (§ risk model) regardless of size.
- Every write carries an explanatory comment with the AI disclaimer
  (`> *This was generated by AI during triage.*`).
- Never leave two open PRs for the same intent (Loop B supersession always closes
  the original once the replacement exists).
- `max_items` bounds blast radius on every run; start a first run scoped by
  `repos`, not org-wide `accounts` (see § Inputs).

## Related skills

- `github-ci-failure-sweep` / `github-issue-tracking` / `github-backlog-planner` —
  Loop A's read-only intake (Step 1).
- `github-pr-review-sweep` — Loop B's read-only intake (Step 1).
- `github-triage-resolver` — existing-item risk classifier (`classify_safe.py`),
  the AI-disclaimer comment convention, and `gh_write.py` (reused here for
  comments — this skill does not duplicate it).
- `spec-generator` / `spec-intake-wizard` / `sdd-implementer` / `spec-verifier`
  (universal-skills, `development`) — the SDD atomics driving Steps 3 and 5.
- `github-issue-sdd-planner` (universal-skills, `development-workflows`) —
  reference DAG shape only (issues→spec), not reused unmodified: it is GitLab-
  tooled and stops before risk-gate/implement/merge.
- Design doc: `reports/spec-driven-remediation-loop-design.md`.
