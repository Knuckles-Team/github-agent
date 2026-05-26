import pytest
import requests
from unittest.mock import MagicMock, patch, AsyncMock
from github_agent.api_client import Api
from github_agent.github_response_models import Release, WorkflowRun


@pytest.fixture
def mock_session():
    with patch("requests.Session") as mock_s:
        session = mock_s.return_value
        response = MagicMock(spec=requests.Response)
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {}
        response.text = "{}"
        session.get.return_value = response
        session.post.return_value = response
        session.put.return_value = response
        session.delete.return_value = response
        session.patch.return_value = response
        yield session


@pytest.mark.usefixtures("mock_session")
def test_release_crud(mock_session):
    api = Api(token="test")

    # Mock for creating a release
    create_resp = MagicMock(spec=requests.Response)
    create_resp.status_code = 201
    create_resp.json.return_value = {
        "id": 12345,
        "tag_name": "v1.0.0",
        "target_commitish": "main",
        "name": "Initial Release",
        "draft": False,
        "prerelease": False,
        "body": "Description of release",
    }
    mock_session.post.return_value = create_resp

    # 1. Create Release
    res = api.create_release(
        owner="Knuckles-Team",
        repo="test-repo",
        tag_name="v1.0.0",
        name="Initial Release",
        body="Description of release",
    )
    assert isinstance(res.data, Release)
    assert res.data.id == 12345
    assert res.data.tag_name == "v1.0.0"
    mock_session.post.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/test-repo/releases",
        json={
            "tag_name": "v1.0.0",
            "draft": False,
            "prerelease": False,
            "name": "Initial Release",
            "body": "Description of release",
        },
        headers=api.headers,
        verify=api.verify,
        proxies=api.proxies,
    )

    # Mock for getting a release
    get_resp = MagicMock(spec=requests.Response)
    get_resp.status_code = 200
    get_resp.json.return_value = create_resp.json.return_value
    mock_session.get.return_value = get_resp

    # 2. Get Release
    res = api.get_release(owner="Knuckles-Team", repo="test-repo", release_id=12345)
    assert isinstance(res.data, Release)
    assert res.data.id == 12345
    mock_session.get.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/test-repo/releases/12345",
        headers=api.headers,
        verify=api.verify,
        proxies=api.proxies,
    )

    # Mock for updating a release
    update_resp = MagicMock(spec=requests.Response)
    update_resp.status_code = 200
    update_resp.json.return_value = create_resp.json.return_value.copy()
    update_resp.json.return_value["name"] = "Updated Release Name"
    mock_session.patch.return_value = update_resp

    # 3. Update Release
    res = api.update_release(
        owner="Knuckles-Team",
        repo="test-repo",
        release_id=12345,
        name="Updated Release Name",
    )
    assert isinstance(res.data, Release)
    assert res.data.name == "Updated Release Name"
    mock_session.patch.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/test-repo/releases/12345",
        json={"name": "Updated Release Name"},
        headers=api.headers,
        verify=api.verify,
        proxies=api.proxies,
    )

    # Mock for deleting a release
    delete_resp = MagicMock(spec=requests.Response)
    delete_resp.status_code = 204
    delete_resp.text = ""
    mock_session.delete.return_value = delete_resp

    # 4. Delete Release
    res = api.delete_release(owner="Knuckles-Team", repo="test-repo", release_id=12345)
    assert res.data == {"status": "deleted"}
    mock_session.delete.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/test-repo/releases/12345",
        headers=api.headers,
        verify=api.verify,
        proxies=api.proxies,
    )


@pytest.mark.usefixtures("mock_session")
def test_actions_operations(mock_session):
    api = Api(token="test")

    # Mock for getting a workflow run
    run_resp = MagicMock(spec=requests.Response)
    run_resp.status_code = 200
    run_resp.json.return_value = {
        "id": 999,
        "name": "CI Run",
        "head_branch": "main",
        "head_sha": "abc123sha",
        "status": "completed",
        "conclusion": "success",
        "event": "push",
    }
    mock_session.get.return_value = run_resp

    # 1. Get workflow run
    res = api.get_workflow_run(owner="Knuckles-Team", repo="test-repo", run_id=999)
    assert isinstance(res.data, WorkflowRun)
    assert res.data.id == 999

    # 2. Rerun workflow run
    rerun_resp = MagicMock(spec=requests.Response)
    rerun_resp.status_code = 201
    mock_session.post.return_value = rerun_resp
    res = api.rerun_workflow_run(owner="Knuckles-Team", repo="test-repo", run_id=999)
    assert res.data == {"status": "rerun_triggered"}
    mock_session.post.assert_any_call(
        url="https://api.github.com/repos/Knuckles-Team/test-repo/actions/runs/999/rerun",
        headers=api.headers,
        verify=api.verify,
        proxies=api.proxies,
    )

    # 3. Cancel workflow run
    cancel_resp = MagicMock(spec=requests.Response)
    cancel_resp.status_code = 202
    mock_session.post.return_value = cancel_resp
    res = api.cancel_workflow_run(owner="Knuckles-Team", repo="test-repo", run_id=999)
    assert res.data == {"status": "cancelled"}
    mock_session.post.assert_any_call(
        url="https://api.github.com/repos/Knuckles-Team/test-repo/actions/runs/999/cancel",
        headers=api.headers,
        verify=api.verify,
        proxies=api.proxies,
    )

    # 4. Delete workflow run
    delete_resp = MagicMock(spec=requests.Response)
    delete_resp.status_code = 204
    mock_session.delete.return_value = delete_resp
    res = api.delete_workflow_run(owner="Knuckles-Team", repo="test-repo", run_id=999)
    assert res.data == {"status": "deleted"}
    mock_session.delete.assert_called_with(
        url="https://api.github.com/repos/Knuckles-Team/test-repo/actions/runs/999",
        headers=api.headers,
        verify=api.verify,
        proxies=api.proxies,
    )


@pytest.mark.usefixtures("mock_session")
def test_search_and_org_retrieval(mock_session):
    api = Api(token="test")

    # 1. Search Repositories mock
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "total_count": 1,
        "incomplete_results": False,
        "items": [{"id": 1, "name": "my-found-repo"}],
    }
    mock_resp.headers = {}
    mock_session.get.return_value = mock_resp

    res = api.search_repositories(q="Knuckles-Team")
    assert res.data.total_count == 1
    assert res.data.items[0]["name"] == "my-found-repo"

    # 2. Get Org Teams mock
    mock_resp_teams = MagicMock(spec=requests.Response)
    mock_resp_teams.status_code = 200
    mock_resp_teams.json.return_value = [{"id": 1, "name": "Admin Team"}]
    mock_session.get.return_value = mock_resp_teams

    res = api.get_org_teams(org="Knuckles-Team")
    assert isinstance(res.data, list)
    assert res.data[0]["name"] == "Admin Team"


# --- MCP Server Full Coverage Test Suite ---


class AsyncMockContext:
    def __init__(self):
        self.info_calls = []
        self.error_calls = []

    async def info(self, msg):
        self.info_calls.append(msg)

    async def error(self, msg):
        self.error_calls.append(msg)


def create_mock_client():
    client = MagicMock()
    # Mock repositories
    mock_repo = MagicMock()
    mock_repo.model_dump.return_value = {"id": 1, "name": "test"}
    client.get_repositories.return_value = MagicMock(data=[mock_repo])
    client.get_repository.return_value = MagicMock(data=mock_repo)
    client.create_repository.return_value = MagicMock(data=mock_repo)
    client.delete_repository.return_value = MagicMock(data={"status": "deleted"})
    client.update_repository.return_value = MagicMock(data=mock_repo)

    # Mock issues
    mock_issue = MagicMock()
    mock_issue.model_dump.return_value = {"id": 1, "title": "test"}
    client.get_issues.return_value = MagicMock(data=[mock_issue])
    client.get_issue.return_value = MagicMock(data=mock_issue)
    client.create_issue.return_value = MagicMock(data=mock_issue)
    client.update_issue.return_value = MagicMock(data=mock_issue)

    # Mock pulls
    mock_pr = MagicMock()
    mock_pr.model_dump.return_value = {"id": 1, "title": "test"}
    client.get_pull_requests.return_value = MagicMock(data=[mock_pr])
    client.get_pull_request.return_value = MagicMock(data=mock_pr)
    client.create_pull_request.return_value = MagicMock(data=mock_pr)
    client.update_pull_request.return_value = MagicMock(data=mock_pr)

    # Mock contents
    mock_content = MagicMock()
    mock_content.model_dump.return_value = {"path": "README.md"}
    client.get_contents.return_value = MagicMock(data=[mock_content])
    client.create_content.return_value = MagicMock(data=mock_content)
    client.update_content.return_value = MagicMock(data=mock_content)
    client.delete_content.return_value = MagicMock(data={"status": "deleted"})

    # Mock branches
    mock_branch = MagicMock()
    mock_branch.model_dump.return_value = {"name": "main"}
    client.get_branches.return_value = MagicMock(data=[mock_branch])
    client.get_branch.return_value = MagicMock(data=mock_branch)
    client.create_branch.return_value = MagicMock(data={"status": "created"})
    client.delete_branch.return_value = MagicMock(data={"status": "deleted"})

    # Mock commits
    mock_commit = MagicMock()
    mock_commit.model_dump.return_value = {"sha": "abc123sha"}
    client.get_commits.return_value = MagicMock(data=[mock_commit])
    client.get_commit.return_value = MagicMock(data=mock_commit)

    # Mock search
    mock_search = MagicMock()
    mock_search.model_dump.return_value = {"items": []}
    client.search_repositories.return_value = MagicMock(data=mock_search)
    client.search_issues.return_value = MagicMock(data=mock_search)
    client.search_code.return_value = MagicMock(data=mock_search)

    # Mock orgs
    client.get_org_repos.return_value = MagicMock(data=[mock_repo])
    mock_member = MagicMock()
    mock_member.model_dump.return_value = {"login": "test"}
    client.get_org_members.return_value = MagicMock(data=[mock_member])
    client.get_org_teams.return_value = MagicMock(data=[{"id": 1}])

    # Mock collaborators
    mock_c = MagicMock()
    mock_c.model_dump.return_value = {"login": "test"}
    client.get_collaborators.return_value = MagicMock(data=[mock_c])
    client.add_collaborator.return_value = MagicMock(data={"status": "added"})
    client.remove_collaborator.return_value = MagicMock(data={"status": "removed"})

    # Mock actions
    mock_workflow = MagicMock()
    mock_workflow.model_dump.return_value = {"id": 1}
    client.get_workflows.return_value = MagicMock(data=[mock_workflow])
    mock_run = MagicMock()
    mock_run.model_dump.return_value = {"id": 1}
    client.get_workflow_runs.return_value = MagicMock(data=[mock_run])
    client.get_workflow_run.return_value = MagicMock(data=mock_run)
    client.trigger_workflow_dispatch.return_value = MagicMock(
        data={"status": "dispatched"}
    )
    client.rerun_workflow_run.return_value = MagicMock(data={"status": "rerun"})
    client.cancel_workflow_run.return_value = MagicMock(data={"status": "cancelled"})
    client.delete_workflow_run.return_value = MagicMock(data={"status": "deleted"})

    # Mock releases
    mock_release = MagicMock()
    mock_release.model_dump.return_value = {"id": 1}
    client.get_releases.return_value = MagicMock(data=[mock_release])
    client.get_release.return_value = MagicMock(data=mock_release)
    client.create_release.return_value = MagicMock(data=mock_release)
    client.update_release.return_value = MagicMock(data=mock_release)
    client.delete_release.return_value = MagicMock(data={"status": "deleted"})

    return client


async def get_registered_tools():
    from github_agent.mcp_server import get_mcp_instance

    mcp_data = get_mcp_instance()
    mcp = mcp_data[0] if isinstance(mcp_data, tuple) else mcp_data
    # FastMCP list_tools can be sync or async
    import inspect

    if inspect.iscoroutinefunction(mcp.list_tools):
        tools = await mcp.list_tools()
    else:
        tools = mcp.list_tools()
    return {t.name: t.fn for t in tools}


@pytest.mark.anyio
async def test_mcp_repos():
    tools = await get_registered_tools()
    github_repos = tools["github_repos"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    # list
    res = await github_repos(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200
    assert len(res["data"]) == 1

    # get success
    res = await github_repos(
        action="get", params_json='{"owner": "o", "repo": "r"}', client=client, ctx=ctx
    )
    assert res["status"] == 200

    # get missing params
    res = await github_repos(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    # create success
    res = await github_repos(
        action="create", params_json='{"name": "newrepo"}', client=client, ctx=ctx
    )
    assert res["status"] == 201

    # create missing param
    res = await github_repos(action="create", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    # delete success
    res = await github_repos(
        action="delete",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    # delete missing param
    res = await github_repos(action="delete", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    # update success
    res = await github_repos(
        action="update",
        params_json='{"owner": "o", "repo": "r", "description": "d"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    # update missing param
    res = await github_repos(action="update", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    # invalid action
    res = await github_repos(action="invalid", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    # invalid json
    res = await github_repos(
        action="list", params_json="invalid json", client=client, ctx=ctx
    )
    assert res["status"] == 400

    # exception handling
    client.get_repositories.side_effect = Exception("API error")
    res = await github_repos(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 500


@pytest.mark.anyio
async def test_mcp_issues():
    tools = await get_registered_tools()
    github_issues = tools["github_issues"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_issues(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_issues(
        action="get",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_issues(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_issues(
        action="create",
        params_json='{"owner": "o", "repo": "r", "title": "t"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_issues(action="create", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_issues(
        action="update",
        params_json='{"owner": "o", "repo": "r", "number": 1, "state": "closed"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_issues(action="update", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_issues(
        action="invalid", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_pulls():
    tools = await get_registered_tools()
    github_pulls = tools["github_pulls"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_pulls(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_pulls(
        action="get",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_pulls(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_pulls(
        action="create",
        params_json='{"owner": "o", "repo": "r", "title": "t", "head": "h", "base": "b"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_pulls(action="create", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_pulls(
        action="update",
        params_json='{"owner": "o", "repo": "r", "number": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_pulls(action="update", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_pulls(action="invalid", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_contents():
    tools = await get_registered_tools()
    github_contents = tools["github_contents"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    # get list
    res = await github_contents(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    # get single item
    mock_single_content = MagicMock()
    mock_single_content.model_dump.return_value = {"path": "README.md"}
    client.get_contents.return_value = MagicMock(data=mock_single_content)
    res = await github_contents(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_contents(
        action="create",
        params_json='{"owner": "o", "repo": "r", "path": "p", "message": "m", "content": "c"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_contents(
        action="create", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_contents(
        action="update",
        params_json='{"owner": "o", "repo": "r", "path": "p", "message": "m", "content": "c", "sha": "s"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_contents(
        action="update", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_contents(
        action="delete",
        params_json='{"owner": "o", "repo": "r", "path": "p", "message": "m", "sha": "s"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_contents(
        action="delete", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_contents(
        action="invalid", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_branches():
    tools = await get_registered_tools()
    github_branches = tools["github_branches"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_branches(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_branches(
        action="get",
        params_json='{"owner": "o", "repo": "r", "branch": "b"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_branches(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_branches(
        action="create",
        params_json='{"owner": "o", "repo": "r", "branch": "b", "ref": "rf"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_branches(
        action="create", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_branches(
        action="delete",
        params_json='{"owner": "o", "repo": "r", "branch": "b"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_branches(
        action="delete", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_branches(
        action="invalid", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_commits():
    tools = await get_registered_tools()
    github_commits = tools["github_commits"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_commits(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_commits(
        action="get",
        params_json='{"owner": "o", "repo": "r", "sha": "s"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_commits(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_commits(
        action="invalid", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_search():
    tools = await get_registered_tools()
    github_search = tools["github_search"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_search(
        action="repositories", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 200

    res = await github_search(action="issues", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_search(action="code", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_search(
        action="invalid", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_orgs():
    tools = await get_registered_tools()
    github_orgs = tools["github_orgs"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_orgs(action="repos", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_orgs(action="members", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 200

    res = await github_orgs(
        action="teams", params_json='{"org": "o"}', client=client, ctx=ctx
    )
    assert res["status"] == 200

    res = await github_orgs(action="teams", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_orgs(action="invalid", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_collaborators():
    tools = await get_registered_tools()
    github_collaborators = tools["github_collaborators"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_collaborators(
        action="list", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 200

    res = await github_collaborators(
        action="add",
        params_json='{"owner": "o", "repo": "r", "username": "u"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_collaborators(
        action="add", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_collaborators(
        action="remove",
        params_json='{"owner": "o", "repo": "r", "username": "u"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_collaborators(
        action="remove", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_collaborators(
        action="invalid", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_actions():
    tools = await get_registered_tools()
    github_actions = tools["github_actions"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_actions(
        action="list_workflows",
        params_json='{"owner": "o", "repo": "r"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="list_workflows", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_actions(
        action="list_runs", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 200

    res = await github_actions(
        action="get_run",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="get_run", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_actions(
        action="trigger_dispatch",
        params_json='{"owner": "o", "repo": "r", "workflow_id": 1, "ref": "m"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="trigger_dispatch", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_actions(
        action="rerun",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(action="rerun", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_actions(
        action="cancel",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="cancel", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_actions(
        action="delete_run",
        params_json='{"owner": "o", "repo": "r", "run_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_actions(
        action="delete_run", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_actions(
        action="invalid", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_mcp_releases():
    tools = await get_registered_tools()
    github_releases = tools["github_releases"]
    client = create_mock_client()
    ctx = AsyncMockContext()

    res = await github_releases(
        action="list", params_json='{"owner": "o", "repo": "r"}', client=client, ctx=ctx
    )
    assert res["status"] == 200

    res = await github_releases(action="list", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_releases(
        action="get",
        params_json='{"owner": "o", "repo": "r", "release_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_releases(action="get", params_json="{}", client=client, ctx=ctx)
    assert res["status"] == 400

    res = await github_releases(
        action="create",
        params_json='{"owner": "o", "repo": "r", "tag_name": "t"}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 201

    res = await github_releases(
        action="create", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_releases(
        action="update",
        params_json='{"owner": "o", "repo": "r", "release_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_releases(
        action="update", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_releases(
        action="delete",
        params_json='{"owner": "o", "repo": "r", "release_id": 1}',
        client=client,
        ctx=ctx,
    )
    assert res["status"] == 200

    res = await github_releases(
        action="delete", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400

    res = await github_releases(
        action="invalid", params_json="{}", client=client, ctx=ctx
    )
    assert res["status"] == 400


@pytest.mark.anyio
async def test_all_tools_invalid_json():
    tools = await get_registered_tools()
    client = create_mock_client()
    ctx = AsyncMockContext()

    # We test every tool in the registry with invalid JSON
    for tool_name, tool_fn in tools.items():
        res = await tool_fn(
            action="invalid_action", params_json="{", client=client, ctx=ctx
        )
        assert res["status"] == 400
        assert "Invalid params_json" in res["error"]


def test_mcp_server_entrypoint():
    from github_agent.mcp_server import mcp_server

    # Test transport stdio
    with patch("github_agent.mcp_server.get_mcp_instance") as mock_get_instance:
        mock_mcp = MagicMock()
        mock_args = MagicMock()
        mock_args.transport = "stdio"
        mock_get_instance.return_value = (mock_mcp, mock_args, [], [], [])

        mcp_server()
        mock_mcp.run.assert_called_once_with(transport="stdio")

    # Test transport streamable-http
    with patch("github_agent.mcp_server.get_mcp_instance") as mock_get_instance:
        mock_mcp = MagicMock()
        mock_args = MagicMock()
        mock_args.transport = "streamable-http"
        mock_args.host = "127.0.0.1"
        mock_args.port = 8000
        mock_get_instance.return_value = (mock_mcp, mock_args, [], [], [])

        mcp_server()
        mock_mcp.run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000
        )

    # Test transport unsupported
    with (
        patch("github_agent.mcp_server.get_mcp_instance") as mock_get_instance,
        patch("sys.exit") as mock_exit,
    ):
        mock_mcp = MagicMock()
        mock_args = MagicMock()
        mock_args.transport = "unsupported"
        mock_get_instance.return_value = (mock_mcp, mock_args, [], [], [])

        mcp_server()
        mock_exit.assert_called_once_with(1)


def test_requests_dependency_warning_import_error():
    import sys
    import importlib

    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "requests.exceptions":
            raise ImportError("Mocked import error for requests.exceptions")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        import github_agent.mcp_server

        importlib.reload(github_agent.mcp_server)


@pytest.mark.anyio
async def test_mcp_tool_safety_exceptions():
    import json

    tools = await get_registered_tools()
    ctx = AsyncMockContext()

    class FailingClient:
        def __getattr__(self, name):
            def failing_method(*args, **kwargs):
                raise RuntimeError("Simulated safety failure")

            return failing_method

    failing_client = FailingClient()

    valid_actions = {
        "github_branches": "list",
        "github_commits": "list",
        "github_contents": "get",
        "github_issues": "list",
        "github_pulls": "list",
        "github_repos": "list",
        "github_releases": "list",
        "github_actions": "list_workflows",
        "github_orgs": "teams",
        "github_collaborators": "list",
        "github_search": "issues",
    }

    params_data = {
        "owner": "test-owner",
        "repo": "test-repo",
        "path": "test.txt",
        "run_id": 1,
        "org": "test-org",
        "q": "test-query",
    }
    params_json = json.dumps(params_data)

    for tool_name, tool_fn in tools.items():
        action = valid_actions.get(tool_name, "list")
        res = await tool_fn(
            action=action, params_json=params_json, client=failing_client, ctx=ctx
        )
        assert res["status"] == 500
        assert "Simulated safety failure" in res["error"]
