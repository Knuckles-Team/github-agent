import pytest
from github_agent.github_input_models import (
    RepoModel,
    IssueModel,
    PullRequestModel,
    ContentModel,
    BranchModel,
    CommitModel,
    SearchModel,
    OrgRepoModel,
    OrgMemberModel,
    CollaboratorModel,
    WorkflowRunModel,
)


def test_repo_model_fields():
    model = RepoModel(visibility="public", affiliation="owner", type="public")
    assert model.api_parameters["visibility"] == "public"
    assert model.api_parameters["affiliation"] == "owner"
    assert model.api_parameters["type"] == "public"


def test_issue_model_fields():
    # Test list labels
    model1 = IssueModel(
        owner="o",
        repo="r",
        state="open",
        labels=["bug", "help"],
        assignee="a",
        since="2026-05-22",
    )
    assert model1.api_parameters["state"] == "open"
    assert model1.api_parameters["labels"] == "bug,help"
    assert model1.api_parameters["assignee"] == "a"
    assert model1.api_parameters["since"] == "2026-05-22"

    # Test str labels
    model2 = IssueModel(owner="o", repo="r", labels="bug")
    assert model2.api_parameters["labels"] == "bug"


def test_pull_request_model_fields():
    model = PullRequestModel(
        owner="o",
        repo="r",
        state="open",
        head="user:branch",
        base="main",
        sort="created",
        direction="asc",
    )
    assert model.api_parameters["state"] == "open"
    assert model.api_parameters["head"] == "user:branch"
    assert model.api_parameters["base"] == "main"
    assert model.api_parameters["sort"] == "created"
    assert model.api_parameters["direction"] == "asc"


def test_content_model_fields():
    model = ContentModel(owner="o", repo="r", path="p", ref="main")
    assert model.api_parameters["ref"] == "main"


def test_branch_model_fields():
    model = BranchModel(owner="o", repo="r", branch="main")
    # BranchModel is instantiated but branch doesn't modify api_parameters inside model_post_init in code,
    # let's assert default params.
    assert model.api_parameters["page"] == 1


def test_commit_model_fields():
    model = CommitModel(
        owner="o", repo="r", sha="abc", path="p", author="a", since="2026", until="2027"
    )
    assert model.api_parameters["sha"] == "abc"
    assert model.api_parameters["path"] == "p"
    assert model.api_parameters["author"] == "a"
    assert model.api_parameters["since"] == "2026"
    assert model.api_parameters["until"] == "2027"


def test_search_model_fields():
    model = SearchModel(q="test", sort="stars", order="desc")
    assert model.api_parameters["q"] == "test"
    assert model.api_parameters["sort"] == "stars"
    assert model.api_parameters["order"] == "desc"


def test_org_repo_model_fields():
    model = OrgRepoModel(org="o", type="public")
    assert model.api_parameters["type"] == "public"


def test_org_member_model_fields():
    model = OrgMemberModel(org="o", role="admin")
    assert model.api_parameters["role"] == "admin"


def test_collaborator_model_fields():
    model = CollaboratorModel(owner="o", repo="r", affiliation="outside")
    assert model.api_parameters["affiliation"] == "outside"


def test_workflow_run_model_fields():
    model = WorkflowRunModel(owner="o", repo="r", status="completed", branch="main")
    assert model.api_parameters["status"] == "completed"
    assert model.api_parameters["branch"] == "main"
