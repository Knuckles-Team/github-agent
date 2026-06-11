#!/usr/bin/python

from datetime import datetime
from typing import Any

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    HttpUrl,
)


class Response(BaseModel):
    """
    Standard Response Wrapper.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    response: requests.Response
    data: Any | None = None


class User(BaseModel):
    model_config = ConfigDict(extra="allow")
    login: str
    id: int
    node_id: str
    avatar_url: HttpUrl
    url: HttpUrl
    html_url: HttpUrl
    type: str
    site_admin: bool


class Repository(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    node_id: str
    name: str
    full_name: str
    private: bool
    owner: User
    html_url: HttpUrl
    description: str | None = None
    fork: bool
    url: HttpUrl
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime
    git_url: str
    ssh_url: str
    clone_url: HttpUrl
    svn_url: HttpUrl
    homepage: str | None = None
    size: int
    stargazers_count: int
    watchers_count: int
    language: str | None = None
    has_issues: bool
    has_projects: bool
    has_downloads: bool
    has_wiki: bool
    has_pages: bool
    forks_count: int
    mirror_url: HttpUrl | None = None
    archived: bool
    disabled: bool
    open_issues_count: int
    license: dict | None = None
    allow_forking: bool
    is_template: bool
    topics: list[str]
    visibility: str
    forks: int
    open_issues: int
    watchers: int
    default_branch: str


class OrganizationSummary(BaseModel):
    """Organization record as returned by the list endpoints
    (GET /user/orgs and GET /organizations)."""

    model_config = ConfigDict(extra="allow")
    login: str
    id: int
    node_id: str | None = None
    url: HttpUrl | None = None
    avatar_url: HttpUrl | None = None
    description: str | None = None


class Organization(OrganizationSummary):
    """Full organization profile as returned by GET /orgs/{org} and
    PATCH /orgs/{org} (which echoes the updated organization)."""

    name: str | None = None
    company: str | None = None
    blog: str | None = None
    location: str | None = None
    email: str | None = None
    twitter_username: str | None = None
    billing_email: str | None = None
    html_url: HttpUrl | None = None
    public_repos: int | None = None
    followers: int | None = None
    following: int | None = None
    has_organization_projects: bool | None = None
    has_repository_projects: bool | None = None
    default_repository_permission: str | None = None
    members_can_create_repositories: bool | None = None
    web_commit_signoff_required: bool | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class OrganizationMembership(BaseModel):
    """Membership record returned by /orgs/{org}/memberships/{username}."""

    model_config = ConfigDict(extra="allow")
    url: HttpUrl | None = None
    state: str
    role: str
    organization_url: HttpUrl | None = None
    organization: OrganizationSummary | None = None
    user: User | None = None


class Issue(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: HttpUrl
    repository_url: HttpUrl
    labels_url: str
    comments_url: HttpUrl
    events_url: HttpUrl
    html_url: HttpUrl
    id: int
    node_id: str
    number: int
    title: str
    user: User
    labels: list[dict]
    state: str
    locked: bool
    assignee: User | None = None
    assignees: list[User]
    milestone: dict | None = None
    comments: int
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    author_association: str
    active_lock_reason: str | None = None
    body: str | None = None
    reactions: dict | None = None
    timeline_url: HttpUrl
    performed_via_github_app: dict | None = None
    state_reason: str | None = None


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: HttpUrl
    id: int
    node_id: str
    html_url: HttpUrl
    diff_url: HttpUrl
    patch_url: HttpUrl
    issue_url: HttpUrl
    number: int
    state: str
    locked: bool
    title: str
    user: User
    body: str | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    merged_at: datetime | None = None
    merge_commit_sha: str | None = None
    assignee: User | None = None
    assignees: list[User]
    requested_reviewers: list[User]
    requested_teams: list[dict]
    labels: list[dict]
    milestone: dict | None = None
    draft: bool
    commits_url: HttpUrl
    review_comments_url: HttpUrl
    review_comment_url: str
    comments_url: HttpUrl
    statuses_url: HttpUrl
    head: dict
    base: dict
    _links: dict
    author_association: str
    auto_merge: dict | None = None
    active_lock_reason: str | None = None


class Content(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    path: str
    sha: str
    size: int
    url: HttpUrl
    html_url: HttpUrl
    git_url: HttpUrl
    download_url: HttpUrl | None = None
    type: str
    content: str | None = None
    encoding: str | None = None
    _links: dict


class Branch(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    commit: dict
    protected: bool


class Commit(BaseModel):
    model_config = ConfigDict(extra="allow")
    sha: str
    node_id: str
    commit: dict
    url: HttpUrl
    html_url: HttpUrl
    comments_url: HttpUrl
    author: User | None = None
    committer: User | None = None
    parents: list[dict]


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_count: int
    incomplete_results: bool
    items: list[dict]


class Collaborator(BaseModel):
    model_config = ConfigDict(extra="allow")
    login: str
    id: int
    permissions: dict | None = None


class Workflow(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    node_id: str
    name: str
    path: str
    state: str


class WorkflowRun(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    name: str | None = None
    head_branch: str
    head_sha: str
    status: str
    conclusion: str | None = None
    event: str


class Release(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    tag_name: str
    target_commitish: str
    name: str | None = None
    draft: bool
    prerelease: bool
    body: str | None = None


class PagesSource(BaseModel):
    """Publishing source of a legacy-built GitHub Pages site."""

    model_config = ConfigDict(extra="allow")
    branch: str
    path: str = "/"


class PagesSite(BaseModel):
    """GitHub Pages site configuration as returned by
    GET/POST /repos/{owner}/{repo}/pages."""

    model_config = ConfigDict(extra="allow")
    url: HttpUrl | None = None
    status: str | None = None
    cname: str | None = None
    custom_404: bool | None = None
    html_url: HttpUrl | None = None
    build_type: str | None = None
    source: PagesSource | None = None
    public: bool | None = None
    https_enforced: bool | None = None


class PagesNotEnabled(BaseModel):
    """Typed result for GET /repos/{owner}/{repo}/pages responding 404 —
    the repository has no GitHub Pages site."""

    enabled: bool = False
    message: str = (
        "GitHub Pages is not enabled for this repository — enable it with "
        "create_pages / the 'pages_create' action."
    )


class PagesAlreadyEnabled(BaseModel):
    """Typed result for POST /repos/{owner}/{repo}/pages responding
    409 Conflict — a GitHub Pages site already exists."""

    already_enabled: bool = True
    message: str = (
        "GitHub Pages is already enabled for this repository — change its "
        "configuration with update_pages / the 'pages_update' action."
    )


class PagesBuild(BaseModel):
    """A GitHub Pages build record from /repos/{owner}/{repo}/pages/builds."""

    model_config = ConfigDict(extra="allow")
    url: HttpUrl | None = None
    status: str | None = None
    error: dict | None = None
    pusher: User | None = None
    commit: str | None = None
    duration: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PagesBuildRequest(BaseModel):
    """Acknowledgement returned by POST /repos/{owner}/{repo}/pages/builds."""

    model_config = ConfigDict(extra="allow")
    url: HttpUrl | None = None
    status: str | None = None


class CollaboratorInvitation(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    repository: Repository | None = None
    invitee: User | None = None
    inviter: User | None = None
    permissions: str | None = None
    created_at: datetime | str | None = None
    url: HttpUrl | str | None = None
    html_url: HttpUrl | str | None = None
