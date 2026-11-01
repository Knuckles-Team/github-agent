# CONN-GITHUB-AGENT-CONNECTOR-CONTENT — Data model

The connector maps GitHub replies to the SDK change-proposal records. It defines no record of its own.

| GitHub reply | SDK record | Engine class | Key fields |
|---|---|---|---|
| Pull request | `ChangeProposalRef` | `:PullRequest` | number, head commit, base ref, state, draft, additions plus deletions |
| Pull request review | `ReviewRecord` | `:Review` | review ID, reviewer, reviewer kind, state, commit, submitted time |
| Review thread | `ReviewThreadRecord` | `:ReviewThread` | thread node ID, path, line, resolved flag, comment IDs |
| Check run or status context | `CheckRunRecord` | `:CheckRun` | run ID, head commit, name, status, conclusion, URL |
| Webhook delivery | `ScmFeedbackEvent` | `:FeedbackEvent` | delivery ID, kind, proposal, actor, actor kind, time, body digest |

A reviewer login that ends in `[bot]`, or an App author, has the kind `automated`. Every other reviewer has the kind `person`.
