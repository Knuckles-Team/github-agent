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
