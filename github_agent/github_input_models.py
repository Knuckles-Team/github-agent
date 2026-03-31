#!/usr/bin/python

from typing import Union, List, Dict, Optional
from pydantic import (
    BaseModel,
    Field,
)


class BaseModelWrapper(BaseModel):
    """
    Base Model wrapping common functionalities.
    """

    max_pages: Optional[int] = Field(
        description="Max amount of pages to retrieve", default=None
    )
    page: Optional[int] = Field(description="Pagination page", default=1)
    per_page: Optional[int] = Field(description="Results per page", default=100)
    api_parameters: Optional[Dict] = Field(description="API Parameters", default=None)

    def model_post_init(self, __context):
        self.api_parameters = {}
        if self.page:
            self.api_parameters["page"] = self.page
        if self.per_page:
            self.api_parameters["per_page"] = self.per_page


class RepoModel(BaseModelWrapper):
    """
    Pydantic model for Repository requests.
    """

    owner: Optional[str] = Field(
        None, description="The account owner of the repository."
    )
    repo: Optional[str] = Field(None, description="The name of the repository.")
    visibility: Optional[str] = Field(
        None, description="Can be one of all, public, or private."
    )
    affiliation: Optional[str] = Field(None, description="Affiliation filter.")
    type: Optional[str] = Field(
        None, description="Can be one of all, owner, public, private, member."
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.visibility:
            self.api_parameters["visibility"] = self.visibility
        if self.affiliation:
            self.api_parameters["affiliation"] = self.affiliation
        if self.type:
            self.api_parameters["type"] = self.type


class IssueModel(BaseModelWrapper):
    """
    Pydantic model for Issue requests.
    """

    owner: str = Field(..., description="The account owner of the repository.")
    repo: str = Field(..., description="The name of the repository.")
    issue_number: Optional[int] = Field(
        None, description="The number that identifies the issue."
    )
    state: Optional[str] = Field(
        None,
        description="Indicates the state of the issues to return. Can be either open, closed, or all.",
    )
    labels: Optional[Union[str, List[str]]] = Field(
        None, description="A list of comma separated label names."
    )
    assignee: Optional[str] = Field(
        None,
        description="Can be the name of a user. Use none for issues with no assigned user, and * for assigned issues to any user.",
    )
    creator: Optional[str] = Field(None, description="The user that created the issue.")
    mentioned: Optional[str] = Field(
        None, description="A user that is mentioned in the issue."
    )
    since: Optional[str] = Field(
        None, description="Only show notifications updated after the given time."
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.state:
            self.api_parameters["state"] = self.state
        if self.labels:
            if isinstance(self.labels, list):
                self.api_parameters["labels"] = ",".join(self.labels)
            else:
                self.api_parameters["labels"] = self.labels
        if self.assignee:
            self.api_parameters["assignee"] = self.assignee
        if self.since:
            self.api_parameters["since"] = self.since


class PullRequestModel(BaseModelWrapper):
    """
    Pydantic model for Pull Request requests.
    """

    owner: str = Field(..., description="The account owner of the repository.")
    repo: str = Field(..., description="The name of the repository.")
    pull_number: Optional[int] = Field(
        None, description="The number that identifies the pull request."
    )
    state: Optional[str] = Field(
        None, description="State of the PR. (open, closed, or all)"
    )
    head: Optional[str] = Field(
        None,
        description="Filter pulls by head user or head organization and branch name in the format of user:ref-name or organization:ref-name.",
    )
    base: Optional[str] = Field(None, description="Filter pulls by base branch name.")
    sort: Optional[str] = Field(
        None,
        description="What to sort results by. Can be created, updated, popularity, long-running.",
    )
    direction: Optional[str] = Field(
        None, description="The direction of the sort. Can be asc or desc."
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.state:
            self.api_parameters["state"] = self.state
        if self.head:
            self.api_parameters["head"] = self.head
        if self.base:
            self.api_parameters["base"] = self.base
        if self.sort:
            self.api_parameters["sort"] = self.sort
        if self.direction:
            self.api_parameters["direction"] = self.direction


class ContentModel(BaseModelWrapper):
    """
    Pydantic model for Repository Content requests.
    """

    owner: str = Field(..., description="The account owner of the repository.")
    repo: str = Field(..., description="The name of the repository.")
    path: str = Field(..., description="The content path.")
    ref: Optional[str] = Field(
        None,
        description="The name of the commit/branch/tag. Default: the repository's default branch.",
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.ref:
            self.api_parameters["ref"] = self.ref


class BranchModel(BaseModelWrapper):
    """
    Pydantic model for Branch requests.
    """

    owner: str = Field(..., description="The account owner of the repository.")
    repo: str = Field(..., description="The name of the repository.")
    branch: Optional[str] = Field(None, description="The name of the branch.")

    def model_post_init(self, __context):
        super().model_post_init(__context)


class CommitModel(BaseModelWrapper):
    """
    Pydantic model for Commit requests.
    """

    owner: str = Field(..., description="The account owner of the repository.")
    repo: str = Field(..., description="The name of the repository.")
    sha: Optional[str] = Field(
        None, description="SHA or branch to start listing commits from."
    )
    path: Optional[str] = Field(
        None, description="Only commits containing this file path will be returned."
    )
    author: Optional[str] = Field(
        None, description="GitHub username or email address to filter by commit author."
    )
    since: Optional[str] = Field(
        None, description="Only show notifications updated after the given time."
    )
    until: Optional[str] = Field(
        None, description="Only show notifications updated before the given time."
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.sha:
            self.api_parameters["sha"] = self.sha
        if self.path:
            self.api_parameters["path"] = self.path
        if self.author:
            self.api_parameters["author"] = self.author
        if self.since:
            self.api_parameters["since"] = self.since
        if self.until:
            self.api_parameters["until"] = self.until
