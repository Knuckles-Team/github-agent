#!/usr/bin/env python
from github_agent.api.api_client_base import logger  # noqa: F401
from github_agent.api.api_client_branches import Api as BranchesApi
from github_agent.api.api_client_commits import Api as CommitsApi
from github_agent.api.api_client_contents import Api as ContentsApi
from github_agent.api.api_client_dependabot import Api as DependabotApi
from github_agent.api.api_client_issues import Api as IssuesApi
from github_agent.api.api_client_orgs import Api as OrgsApi
from github_agent.api.api_client_pages import Api as PagesApi
from github_agent.api.api_client_pulls import Api as PullsApi
from github_agent.api.api_client_releases import Api as ReleasesApi
from github_agent.api.api_client_repos import Api as ReposApi
from github_agent.api.api_client_search import Api as SearchApi
from github_agent.api.api_client_workflows import Api as WorkflowsApi
from github_agent.github_response_models import (  # noqa: F401
    Branch,
    Collaborator,
    Commit,
    Content,
    Issue,
    Organization,
    OrganizationMembership,
    OrganizationSummary,
    PagesAlreadyEnabled,
    PagesBuild,
    PagesBuildRequest,
    PagesNotEnabled,
    PagesSite,
    PullRequest,
    Release,
    Repository,
    SearchResult,
    User,
    Workflow,
    WorkflowRun,
)


class Api(
    BranchesApi,
    CommitsApi,
    ContentsApi,
    DependabotApi,
    IssuesApi,
    OrgsApi,
    PagesApi,
    PullsApi,
    ReleasesApi,
    ReposApi,
    SearchApi,
    WorkflowsApi,
):
    pass
