import pytest
import requests
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
from agent_utilities.exceptions import ParameterError, MissingParameterError
from github_agent.api_client import Api


# Let's mock all Pydantic response models inside github_agent.api_client so they accept anything and don't raise ValidationError on dummy data
@pytest.fixture(autouse=True)
def mock_response_models():
    mocks = [
        ("github_agent.api_client.Repository", lambda **k: MagicMock()),
        ("github_agent.api.api_client_repos.Repository", lambda **k: MagicMock()),
        ("github_agent.api_client.Commit", lambda **k: MagicMock()),
        ("github_agent.api.api_client_commits.Commit", lambda **k: MagicMock()),
        ("github_agent.api_client.Branch", lambda **k: MagicMock()),
        ("github_agent.api.api_client_branches.Branch", lambda **k: MagicMock()),
        ("github_agent.api_client.Content", lambda **k: MagicMock()),
        ("github_agent.api.api_client_contents.Content", lambda **k: MagicMock()),
        ("github_agent.api_client.Issue", lambda **k: MagicMock()),
        ("github_agent.api.api_client_issues.Issue", lambda **k: MagicMock()),
        ("github_agent.api_client.PullRequest", lambda **k: MagicMock()),
        ("github_agent.api.api_client_pulls.PullRequest", lambda **k: MagicMock()),
        ("github_agent.api_client.SearchResult", lambda **k: MagicMock()),
        ("github_agent.api.api_client_search.SearchResult", lambda **k: MagicMock()),
        ("github_agent.api_client.Collaborator", lambda **k: MagicMock()),
        ("github_agent.api.api_client_repos.Collaborator", lambda **k: MagicMock()),
        ("github_agent.api_client.Workflow", lambda **k: MagicMock()),
        ("github_agent.api.api_client_workflows.Workflow", lambda **k: MagicMock()),
        ("github_agent.api_client.WorkflowRun", lambda **k: MagicMock()),
        ("github_agent.api.api_client_workflows.WorkflowRun", lambda **k: MagicMock()),
        ("github_agent.api_client.Release", lambda **k: MagicMock()),
        ("github_agent.api.api_client_releases.Release", lambda **k: MagicMock()),
        ("github_agent.api_client.User", lambda **k: MagicMock()),
        ("github_agent.api.api_client_orgs.User", lambda **k: MagicMock()),
    ]
    with ExitStack() as stack:
        for target, new_val in mocks:
            stack.enter_context(patch(target, new_val))
        yield


@pytest.fixture
def mock_api_session():
    with patch("requests.Session") as mock_s:
        session = mock_s.return_value

        # Default mock response for /user check on Api init
        mock_user_resp = MagicMock(spec=requests.Response)
        mock_user_resp.status_code = 200
        mock_user_resp.headers = {}
        mock_user_resp.json.return_value = {"id": 1, "login": "test"}

        session.get.return_value = mock_user_resp
        session.post.return_value = mock_user_resp
        session.put.return_value = mock_user_resp
        session.delete.return_value = mock_user_resp
        session.patch.return_value = mock_user_resp

        yield session


def test_get_repository_edge_cases(mock_api_session):
    api = Api(token="test")

    # HTTPError with 404
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 404
    http_err_404 = requests.exceptions.HTTPError("Not Found", response=mock_resp)
    mock_api_session.get.side_effect = http_err_404

    with pytest.raises(ParameterError, match="Repository test/test not found"):
        api.get_repository("test", "test")

    # HTTPError with 500
    mock_resp_500 = MagicMock(spec=requests.Response)
    mock_resp_500.status_code = 500
    http_err_500 = requests.exceptions.HTTPError("Server Error", response=mock_resp_500)
    mock_api_session.get.side_effect = http_err_500

    with pytest.raises(requests.exceptions.HTTPError):
        api.get_repository("test", "test")


def test_missing_parameters(mock_api_session):
    api = Api(token="test")

    # get_issues without owner
    with pytest.raises(MissingParameterError, match="owner and repo are required"):
        api.get_issues(repo="r")

    # get_pull_requests without repo
    with pytest.raises(MissingParameterError, match="owner and repo are required"):
        api.get_pull_requests(owner="o")


def test_get_contents_list_vs_dict(mock_api_session):
    api = Api(token="test")

    # 1. Directory (list response)
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"name": "file1.txt"}]
    mock_api_session.get.return_value = mock_resp

    res = api.get_contents(owner="o", repo="r", path="p")
    assert isinstance(res.data, list)

    # 2. File (dict response)
    mock_resp_dict = MagicMock(spec=requests.Response)
    mock_resp_dict.status_code = 200
    mock_resp_dict.json.return_value = {"name": "file1.txt"}
    mock_api_session.get.return_value = mock_resp_dict

    res2 = api.get_contents(owner="o", repo="r", path="p")
    assert not isinstance(res2.data, list)


def test_pagination_and_thread_pool_error(mock_api_session):
    api = Api(token="test")

    # Mocking first page to have Link header for multiple pages
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.headers = {
        "Link": '<https://api.github.com/user/repos?page=3>; rel="last"'
    }
    mock_resp.json.return_value = [{"id": 1}]

    mock_api_session.get.return_value = mock_resp

    # Mock _fetch_next_page to raise Exception to hit the except block inside threadpool loop
    with patch.object(
        api, "_fetch_next_page", side_effect=Exception("Failed page fetch")
    ):
        res = api.get_repositories()
        # Even with failed thread page fetches, it returns the first page's data
        assert len(res.data) == 1


def test_get_total_pages_variations(mock_api_session):
    api = Api(token="test")

    # 1. No Link header
    mock_resp1 = MagicMock(spec=requests.Response)
    mock_resp1.headers = {}
    assert api._get_total_pages(mock_resp1) == 1

    # 2. Link header without rel="last"
    mock_resp2 = MagicMock(spec=requests.Response)
    mock_resp2.headers = {
        "Link": '<https://api.github.com/user/repos?page=3>; rel="next"'
    }
    assert api._get_total_pages(mock_resp2) == 1

    # 3. Link header with rel="last"
    mock_resp3 = MagicMock(spec=requests.Response)
    mock_resp3.headers = {
        "Link": '<https://api.github.com/user/repos?page=42>; rel="last"'
    }
    assert api._get_total_pages(mock_resp3) == 42
