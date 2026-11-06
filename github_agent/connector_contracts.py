"""Tool-use contracts, surface links, data locations and business roles of github-mcp.

The SDK serves this declaration as the connector's surface ontology and checks it
against the served tools, prompts and skills (SCION-CONNECTOR-CONTROL-R916).
"""

# ruff: noqa: E501
from __future__ import annotations

from scion_sdk.surface_contracts import (
    BusinessRole,
    DataLocation,
    Refusal,
    SurfaceLink,
    ToolStep,
    ToolUseContract,
)

CONTRACTS: tuple[ToolUseContract, ...] = (
    ToolUseContract(
        "find-repositories",
        description="Find repositories and read their details",
        steps=(
            ToolStep(
                "github_get_org_repos",
                params=("org",),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_get_organization",
                params=("org",),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_repositories",
                params=("max_pages", "page", "per_page"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_get_repository",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_list_organizations",
                params=("max_pages", "page", "per_page"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_search_repositories",
                params=("q",),
                result=("result",),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "provision-a-repository",
        description="Create a repository, set branch protection, secrets and Pages",
        steps=(
            ToolStep(
                "github_create_or_update_repo_secret",
                params=("owner", "repo", "secret_name", "encrypted_value", "key_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_create_organization_repository",
                params=("org", "name"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_create_pages",
                params=("owner", "repo"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_create_repository",
                params=("name",),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_get_branch_protection",
                params=("owner", "repo", "branch"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_pages",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_pages_build_latest",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_repo_secret_public_key",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_repo_secrets",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_list_pages_builds",
                params=("owner", "repo"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_request_pages_build",
                params=("owner", "repo"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_update_branch_protection",
                params=("owner", "repo", "branch", "protection_config"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_update_pages",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_update_repository",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
        ),
    ),
    ToolUseContract(
        "retire-a-repository",
        description="Delete a repository, secret, branch protection or Pages site",
        steps=(
            ToolStep(
                "github_delete_branch_protection",
                params=("owner", "repo", "branch"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_delete_pages",
                params=("owner", "repo"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_delete_repo_secret",
                params=("owner", "repo", "secret_name"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_delete_repository",
                params=("owner", "repo"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "work-issues",
        description="Read, open, change and comment on issues",
        steps=(
            ToolStep(
                "github_create_issue",
                params=("owner", "repo", "title"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_create_issue_comment",
                params=("owner", "repo", "issue_number", "body"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_delete_issue_comment",
                params=("owner", "repo", "comment_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_get_issue",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_issue_comment",
                params=("owner", "repo", "comment_id"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_issues",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_list_issue_comments",
                params=("owner", "repo", "issue_number"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_list_repo_issue_comments",
                params=("owner", "repo"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_search_issues", params=("q",), result=("result",), refusals=()
            ),
            ToolStep(
                "github_update_issue",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_update_issue_comment",
                params=("owner", "repo", "comment_id", "body"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
        ),
    ),
    ToolUseContract(
        "create-a-pull-request",
        description="Branch, commit content and open a pull request",
        steps=(
            ToolStep(
                "github_create_branch",
                params=("owner", "repo", "branch", "ref"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_create_content",
                params=("owner", "repo", "path", "message", "content"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_create_pull_request",
                params=("owner", "repo", "title", "head", "base"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_get_branch",
                params=("owner", "repo", "branch"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_branches",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_contents",
                params=("owner", "repo", "path"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_request_reviewers",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_update_content",
                params=("owner", "repo", "path", "message", "content", "sha"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
        ),
    ),
    ToolUseContract(
        "review-a-pull-request",
        description="Read a pull request, review it, comment on lines and resolve comments",
        steps=(
            ToolStep(
                "github_create_commit_comment",
                params=("owner", "repo", "sha", "body"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_create_pull_request_review",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_create_review_comment",
                params=("owner", "repo", "pull_number", "body"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_create_review_comment_reply",
                params=("owner", "repo", "pull_number", "comment_id", "body"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_delete_commit_comment",
                params=("owner", "repo", "comment_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_delete_review_comment",
                params=("owner", "repo", "comment_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_get_commit",
                params=("owner", "repo", "sha"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_commit_comment",
                params=("owner", "repo", "comment_id"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_commits",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_pull_request",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_pull_requests",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_review_comment",
                params=("owner", "repo", "comment_id"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_list_commit_comments",
                params=("owner", "repo", "sha"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_list_repo_commit_comments",
                params=("owner", "repo"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_list_repo_review_comments",
                params=("owner", "repo"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_list_review_comments",
                params=("owner", "repo", "pull_number"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_update_commit_comment",
                params=("owner", "repo", "comment_id", "body"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_update_review_comment",
                params=("owner", "repo", "comment_id", "body"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
        ),
    ),
    ToolUseContract(
        "land-a-pull-request",
        description="Update, merge or auto-merge a pull request and delete its branch",
        steps=(
            ToolStep(
                "github_delete_branch",
                params=("owner", "repo", "branch"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_merge_pull_request",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_pulls_disable_auto_merge",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_pulls_enable_auto_merge",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_update_pull_request",
                params=("owner", "repo", "number"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
        ),
    ),
    ToolUseContract(
        "review-ci-results",
        description="Read workflow runs, jobs and logs of a repository",
        steps=(
            ToolStep(
                "github_get_workflow_job_logs",
                params=("owner", "repo", "job_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_get_workflow_run",
                params=("owner", "repo", "run_id"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_workflow_run_jobs",
                params=("owner", "repo", "run_id"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_workflow_runs",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_workflows",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_ingest_pipelines",
                params=("owner", "repo"),
                result=("result",),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "run-ci",
        description="Trigger, re-run, cancel and delete workflow runs",
        steps=(
            ToolStep(
                "github_cancel_workflow_run",
                params=("owner", "repo", "run_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_delete_workflow_run",
                params=("owner", "repo", "run_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_rerun_workflow_run",
                params=("owner", "repo", "run_id"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_trigger_workflow_dispatch",
                params=("owner", "repo", "workflow_id", "ref"),
                result=("result",),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "review-dependabot-alerts",
        description="Read and dismiss Dependabot alerts of a repository or organization",
        steps=(
            ToolStep(
                "github_get_dependabot_alert",
                params=("owner", "repo", "alert_number"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_get_dependabot_alerts",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_org_dependabot_alerts",
                params=("org",),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_update_dependabot_alert",
                params=("owner", "repo", "alert_number", "state"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
        ),
    ),
    ToolUseContract(
        "publish-a-release",
        description="Create, change and delete releases",
        steps=(
            ToolStep(
                "github_create_release",
                params=("owner", "repo", "tag_name"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_delete_release",
                params=("owner", "repo", "release_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_get_release",
                params=("owner", "repo", "release_id"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_releases",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_update_release",
                params=("owner", "repo", "release_id"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
        ),
    ),
    ToolUseContract(
        "manage-collaborators",
        description="Add and remove repository collaborators",
        steps=(
            ToolStep(
                "github_add_collaborator",
                params=("owner", "repo", "username"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_get_collaborators",
                params=("owner", "repo"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_remove_collaborator",
                params=("owner", "repo", "username"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "administer-organization",
        description="Manage organization membership, teams and settings",
        steps=(
            ToolStep(
                "github_create_organization",
                params=("login", "admin"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_delete_organization",
                params=("org",),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_get_org_members",
                params=("org",),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_org_teams",
                params=("org",),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_get_organization_membership",
                params=("org", "username"),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
            ToolStep(
                "github_remove_organization_member",
                params=("org", "username"),
                result=("result",),
                refusals=(
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
            ToolStep(
                "github_set_organization_membership",
                params=("org", "username"),
                result=("result",),
                refusals=(),
            ),
            ToolStep(
                "github_update_organization",
                params=("org",),
                result=("result",),
                refusals=(Refusal("not_found", "the named object does not exist"),),
            ),
        ),
    ),
    ToolUseContract(
        "search-code",
        description="Search code across repositories",
        steps=(
            ToolStep(
                "github_search_code",
                params=("q",),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        "query-graphql",
        description="Run a GraphQL query or discover the schema for fleet-wide reads",
        steps=(
            ToolStep(
                "github_discover_graphql_schema",
                params=("type_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_graphql", params=("query",), result=("result",), refusals=()
            ),
        ),
    ),
    ToolUseContract(
        "ingest-repositories",
        description="Mirror repositories into the knowledge graph",
        steps=(
            ToolStep(
                "github_get_repositories",
                params=("max_pages", "page", "per_page"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                ),
            ),
            ToolStep(
                "github_ingest_repos",
                params=("visibility", "affiliation", "type"),
                result=("result",),
                refusals=(),
            ),
        ),
    ),
    ToolUseContract(
        "other-operations",
        description="Operations not covered by a named workflow",
        steps=(
            ToolStep(
                "github_delete_content",
                params=("owner", "repo", "path", "message", "sha"),
                result=("result",),
                refusals=(
                    Refusal(
                        "access_denied", "the credential lacks access to the system"
                    ),
                    Refusal(
                        "confirmation_required",
                        "the destructive call omits confirm=true",
                    ),
                    Refusal("not_found", "the named object does not exist"),
                ),
            ),
        ),
    ),
)

LINKS: tuple[SurfaceLink, ...] = (
    SurfaceLink(
        "find-repositories",
        tools=(
            "github_get_org_repos",
            "github_get_organization",
            "github_get_repositories",
            "github_get_repository",
            "github_list_organizations",
            "github_search_repositories",
        ),
        prompts=(
            "github-mcp-operator",
            "github-mcp-repository-administrator",
            "github-mcp-triage-lead",
        ),
        skills=(
            "github-agent-operations",
            "github-backlog-planner",
            "github-repository-management",
        ),
    ),
    SurfaceLink(
        "provision-a-repository",
        tools=(
            "github_create_or_update_repo_secret",
            "github_create_organization_repository",
            "github_create_pages",
            "github_create_repository",
            "github_get_branch_protection",
            "github_get_pages",
            "github_get_pages_build_latest",
            "github_get_repo_secret_public_key",
            "github_get_repo_secrets",
            "github_list_pages_builds",
            "github_request_pages_build",
            "github_update_branch_protection",
            "github_update_pages",
            "github_update_repository",
        ),
        prompts=("github-mcp-operator", "github-mcp-repository-administrator"),
        skills=("github-project-provisioner", "github-repository-management"),
    ),
    SurfaceLink(
        "retire-a-repository",
        tools=(
            "github_delete_branch_protection",
            "github_delete_pages",
            "github_delete_repo_secret",
            "github_delete_repository",
        ),
        prompts=("github-mcp-operator", "github-mcp-repository-administrator"),
        skills=("github-repository-management",),
    ),
    SurfaceLink(
        "work-issues",
        tools=(
            "github_create_issue",
            "github_create_issue_comment",
            "github_delete_issue_comment",
            "github_get_issue",
            "github_get_issue_comment",
            "github_get_issues",
            "github_list_issue_comments",
            "github_list_repo_issue_comments",
            "github_search_issues",
            "github_update_issue",
            "github_update_issue_comment",
        ),
        prompts=("github-mcp-operator", "github-mcp-triage-lead"),
        skills=(
            "github-backlog-planner",
            "github-issue-tracking",
            "github-triage-resolver",
        ),
    ),
    SurfaceLink(
        "create-a-pull-request",
        tools=(
            "github_create_branch",
            "github_create_content",
            "github_create_pull_request",
            "github_get_branch",
            "github_get_branches",
            "github_get_contents",
            "github_request_reviewers",
            "github_update_content",
        ),
        prompts=("github-mcp-operator", "github-mcp-release-manager"),
        skills=(
            "github-org-remediation-loop",
            "github-pull-request-create",
            "github-triage-resolver",
        ),
    ),
    SurfaceLink(
        "review-a-pull-request",
        tools=(
            "github_create_commit_comment",
            "github_create_pull_request_review",
            "github_create_review_comment",
            "github_create_review_comment_reply",
            "github_delete_commit_comment",
            "github_delete_review_comment",
            "github_get_commit",
            "github_get_commit_comment",
            "github_get_commits",
            "github_get_pull_request",
            "github_get_pull_requests",
            "github_get_review_comment",
            "github_list_commit_comments",
            "github_list_repo_commit_comments",
            "github_list_repo_review_comments",
            "github_list_review_comments",
            "github_update_commit_comment",
            "github_update_review_comment",
        ),
        prompts=("github-mcp-operator", "github-mcp-release-manager"),
        skills=("github-pr-review-sweep", "github-pull-request-review"),
    ),
    SurfaceLink(
        "land-a-pull-request",
        tools=(
            "github_delete_branch",
            "github_merge_pull_request",
            "github_pulls_disable_auto_merge",
            "github_pulls_enable_auto_merge",
            "github_update_pull_request",
        ),
        prompts=("github-mcp-operator", "github-mcp-release-manager"),
        skills=(
            "github-org-remediation-loop",
            "github-pr-review-sweep",
            "github-pull-request-review",
        ),
    ),
    SurfaceLink(
        "review-ci-results",
        tools=(
            "github_get_workflow_job_logs",
            "github_get_workflow_run",
            "github_get_workflow_run_jobs",
            "github_get_workflow_runs",
            "github_get_workflows",
            "github_ingest_pipelines",
        ),
        prompts=("github-mcp-operator", "github-mcp-release-manager"),
        skills=(
            "github-actions-ci-review",
            "github-agent-operations",
            "github-ci-failure-sweep",
        ),
    ),
    SurfaceLink(
        "run-ci",
        tools=(
            "github_cancel_workflow_run",
            "github_delete_workflow_run",
            "github_rerun_workflow_run",
            "github_trigger_workflow_dispatch",
        ),
        prompts=("github-mcp-operator", "github-mcp-release-manager"),
        skills=("github-actions-ci-review", "github-ci-failure-sweep"),
    ),
    SurfaceLink(
        "review-dependabot-alerts",
        tools=(
            "github_get_dependabot_alert",
            "github_get_dependabot_alerts",
            "github_get_org_dependabot_alerts",
            "github_update_dependabot_alert",
        ),
        prompts=("github-mcp-operator", "github-mcp-triage-lead"),
        skills=("github-dependabot-review", "github-org-remediation-loop"),
    ),
    SurfaceLink(
        "publish-a-release",
        tools=(
            "github_create_release",
            "github_delete_release",
            "github_get_release",
            "github_get_releases",
            "github_update_release",
        ),
        prompts=("github-mcp-operator", "github-mcp-release-manager"),
        skills=("github-agent-operations", "github-tools"),
    ),
    SurfaceLink(
        "manage-collaborators",
        tools=(
            "github_add_collaborator",
            "github_get_collaborators",
            "github_remove_collaborator",
        ),
        prompts=("github-mcp-operator", "github-mcp-repository-administrator"),
        skills=("github-repository-management",),
    ),
    SurfaceLink(
        "administer-organization",
        tools=(
            "github_create_organization",
            "github_delete_organization",
            "github_get_org_members",
            "github_get_org_teams",
            "github_get_organization_membership",
            "github_remove_organization_member",
            "github_set_organization_membership",
            "github_update_organization",
        ),
        prompts=("github-mcp-operator", "github-mcp-repository-administrator"),
        skills=("github-org-remediation-loop",),
    ),
    SurfaceLink(
        "search-code",
        tools=("github_search_code",),
        prompts=("github-mcp-operator", "github-mcp-triage-lead"),
        skills=("github-tools",),
    ),
    SurfaceLink(
        "query-graphql",
        tools=("github_discover_graphql_schema", "github_graphql"),
        prompts=("github-mcp-operator", "github-mcp-triage-lead"),
        skills=("github-agent-operations", "github-tools"),
    ),
    SurfaceLink(
        "ingest-repositories",
        tools=("github_get_repositories", "github_ingest_repos"),
        prompts=("github-mcp-operator", "github-mcp-triage-lead"),
        skills=("github-tools",),
    ),
    SurfaceLink(
        "other-operations",
        tools=("github_delete_content",),
        prompts=("github-mcp-operator",),
        skills=(),
    ),
)

LOCATIONS: tuple[DataLocation, ...] = (
    DataLocation(
        "github-repositories",
        ("SoftwareProject", "Repository"),
        "GITHUB_URL",
        "github_get_repositories",
        federation_key="repo",
    ),
    DataLocation(
        "github-issues",
        ("Issue",),
        "GITHUB_URL",
        "github_get_issues",
        federation_key="number",
    ),
    DataLocation(
        "github-pull-requests",
        ("PullRequest",),
        "GITHUB_URL",
        "github_get_pull_requests",
        federation_key="number",
    ),
    DataLocation(
        "github-actions-runs",
        ("Deployment",),
        "GITHUB_URL",
        "github_get_workflow_runs",
        federation_key="run_id",
    ),
    DataLocation(
        "github-organizations",
        ("Organization", "Person"),
        "GITHUB_URL",
        "github_get_organization",
        federation_key="org",
    ),
)

ROLES: tuple[BusinessRole, ...] = (
    BusinessRole(
        "release-manager",
        tasks=(
            "create-a-pull-request",
            "review-a-pull-request",
            "land-a-pull-request",
            "review-ci-results",
            "run-ci",
            "publish-a-release",
        ),
    ),
    BusinessRole(
        "repository-administrator",
        tasks=(
            "find-repositories",
            "provision-a-repository",
            "retire-a-repository",
            "manage-collaborators",
            "administer-organization",
        ),
    ),
    BusinessRole(
        "triage-lead",
        tasks=(
            "work-issues",
            "review-dependabot-alerts",
            "search-code",
            "query-graphql",
            "ingest-repositories",
            "find-repositories",
        ),
    ),
)
