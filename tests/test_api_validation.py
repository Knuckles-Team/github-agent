from typing import Any
import pytest
from unittest.mock import MagicMock, patch
import requests
from agent_utilities.exceptions import ParameterError
from github_agent.api_client import Api


def test_all_api_validation_errors():
    with patch("requests.Session") as mock_s:
        session = mock_s.return_value

        # Mock response for /user check on Api init
        mock_user_resp = MagicMock(spec=requests.Response)
        mock_user_resp.status_code = 200
        mock_user_resp.json.return_value = {"id": 1, "login": "test"}
        session.get.return_value = mock_user_resp

        api = Api(token="test")

        # Now change the return values of session methods to return empty dicts/lists to fail Pydantic validation
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        session.get.return_value = mock_resp
        session.post.return_value = mock_resp
        session.put.return_value = mock_resp
        session.delete.return_value = mock_resp
        session.patch.return_value = mock_resp

        # Also mock _fetch_all_pages to return invalid data [{}]
        setattr(api, "_fetch_all_pages", MagicMock(return_value=(mock_resp, [{}])))

        # List of API methods to test with validation error
        api_methods: list[tuple[str, dict[str, Any]]] = [
            ("get_repositories", {}),
            ("get_issues", {"owner": "o", "repo": "r"}),
            ("get_pull_requests", {"owner": "o", "repo": "r"}),
            ("get_branches", {"owner": "o", "repo": "r"}),
            ("get_commits", {"owner": "o", "repo": "r"}),
            ("create_repository", {"name": "n"}),
            ("update_repository", {"owner": "o", "repo": "r"}),
            ("get_collaborators", {"owner": "o", "repo": "r"}),
            ("add_collaborator", {"owner": "o", "repo": "r", "username": "u"}),
            ("get_contents", {"owner": "o", "repo": "r", "path": "p"}),
            (
                "create_content",
                {
                    "owner": "o",
                    "repo": "r",
                    "path": "p",
                    "message": "m",
                    "content": "c",
                },
            ),
            (
                "update_content",
                {
                    "owner": "o",
                    "repo": "r",
                    "path": "p",
                    "message": "m",
                    "content": "c",
                    "sha": "s",
                },
            ),
            ("get_workflows", {"owner": "o", "repo": "r"}),
            ("get_workflow_runs", {"owner": "o", "repo": "r"}),
            ("get_workflow_run", {"owner": "o", "repo": "r", "run_id": 123}),
            ("get_releases", {"owner": "o", "repo": "r"}),
            ("get_release", {"owner": "o", "repo": "r", "release_id": 123}),
            ("create_release", {"owner": "o", "repo": "r", "tag_name": "t"}),
            ("update_release", {"owner": "o", "repo": "r", "release_id": 123}),
        ]

        for method_name, kwargs in api_methods:
            print(f"Testing {method_name}...")
            method = getattr(api, method_name)
            with pytest.raises(ParameterError, match="Invalid parameters"):
                method(**kwargs)
