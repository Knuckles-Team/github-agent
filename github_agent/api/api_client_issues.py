#!/usr/bin/env python
from agent_utilities.decorators import require_auth
from agent_utilities.exceptions import (
    MissingParameterError,
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    IssueModel,
)
from github_agent.github_response_models import (
    Issue,
    Response,
)


class Api(BaseApiClient):
    @require_auth
    def get_issues(self, **kwargs) -> Response:
        """List issues for a repository."""
        if not kwargs.get("owner") or not kwargs.get("repo"):
            raise MissingParameterError("owner and repo are required")
        model = IssueModel(**kwargs)
        try:
            response, data = self._fetch_all_pages(
                f"/repos/{model.owner}/{model.repo}/issues", model
            )
            parsed_data = [Issue(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_issue(self, owner: str, repo: str, title: str, **kwargs) -> Response:
        """Create a new issue in a repository."""
        try:
            payload = {"title": title, **kwargs}
            response = self._session.post(
                url=f"{self.url}/repos/{owner}/{repo}/issues",
                json=payload,
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Issue(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def update_issue(self, owner: str, repo: str, number: int, **kwargs) -> Response:
        """Update an issue in a repository."""
        try:
            response = self._session.patch(
                url=f"{self.url}/repos/{owner}/{repo}/issues/{number}",
                json=kwargs,
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Issue(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_issue(self, owner: str, repo: str, number: int) -> Response:
        """Get a single issue in a repository."""
        try:
            response = self._session.get(
                url=f"{self.url}/repos/{owner}/{repo}/issues/{number}",
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Issue(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
