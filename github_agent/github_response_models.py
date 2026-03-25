#!/usr/bin/python
# coding: utf-8
from typing import List, Dict, Optional, Any
from pydantic import (
    BaseModel,
    ConfigDict,
    HttpUrl,
)
from datetime import datetime
import requests


class Response(BaseModel):
    """
    Standard Response Wrapper.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    response: requests.Response
    data: Any = None


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
    description: Optional[str] = None
    fork: bool
    url: HttpUrl
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime
    git_url: str
    ssh_url: str
    clone_url: HttpUrl
    svn_url: HttpUrl
    homepage: Optional[str] = None
    size: int
    stargazers_count: int
    watchers_count: int
    language: Optional[str] = None
    has_issues: bool
    has_projects: bool
    has_downloads: bool
    has_wiki: bool
    has_pages: bool
    forks_count: int
    mirror_url: Optional[HttpUrl] = None
    archived: bool
    disabled: bool
    open_issues_count: int
    license: Optional[Dict] = None
    allow_forking: bool
    is_template: bool
    topics: List[str]
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
    labels: List[Dict]
    state: str
    locked: bool
    assignee: Optional[User] = None
    assignees: List[User]
    milestone: Optional[Dict] = None
    comments: int
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None
    author_association: str
    active_lock_reason: Optional[str] = None
    body: Optional[str] = None
    reactions: Optional[Dict] = None
    timeline_url: HttpUrl
    performed_via_github_app: Optional[Dict] = None
    state_reason: Optional[str] = None


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
    body: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    merge_commit_sha: Optional[str] = None
    assignee: Optional[User] = None
    assignees: List[User]
    requested_reviewers: List[User]
    requested_teams: List[Dict]
    labels: List[Dict]
    milestone: Optional[Dict] = None
    draft: bool
    commits_url: HttpUrl
    review_comments_url: HttpUrl
    review_comment_url: str
    comments_url: HttpUrl
    statuses_url: HttpUrl
    head: Dict
    base: Dict
    _links: Dict
    author_association: str
    auto_merge: Optional[Dict] = None
    active_lock_reason: Optional[str] = None


class Content(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    path: str
    sha: str
    size: int
    url: HttpUrl
    html_url: HttpUrl
    git_url: HttpUrl
    download_url: Optional[HttpUrl] = None
    type: str
    content: Optional[str] = None
    encoding: Optional[str] = None
    _links: Dict


class Branch(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    commit: Dict
    protected: bool


class Commit(BaseModel):
    model_config = ConfigDict(extra="allow")
    sha: str
    node_id: str
    commit: Dict
    url: HttpUrl
    html_url: HttpUrl
    comments_url: HttpUrl
    author: Optional[User] = None
    committer: Optional[User] = None
    parents: List[Dict]
