#!/usr/bin/env python
from agent_utilities.decorators import require_auth
from agent_utilities.exceptions import (
    MissingParameterError,
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    PullRequestModel,
)
from github_agent.github_response_models import (
    PullRequest,
    Response,
)


class Api(BaseApiClient):
    @require_auth
    def get_pull_requests(self, **kwargs) -> Response:
        """List pull requests for a repository."""
        if not kwargs.get("owner") or not kwargs.get("repo"):
            raise MissingParameterError("owner and repo are required")
        model = PullRequestModel(**kwargs)
        try:
            response, data = self._fetch_all_pages(
                f"/repos/{model.owner}/{model.repo}/pulls", model
            )
            parsed_data = [PullRequest(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, **kwargs
    ) -> Response:
        """Create a new pull request in a repository."""
        try:
            payload = {"title": title, "head": head, "base": base, **kwargs}
            response = self._session.post(
                url=f"{self.url}/repos/{owner}/{repo}/pulls",
                json=payload,
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = PullRequest(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def update_pull_request(
        self, owner: str, repo: str, number: int, **kwargs
    ) -> Response:
        """Update a pull request."""
        try:
            response = self._session.patch(
                url=f"{self.url}/repos/{owner}/{repo}/pulls/{number}",
                json=kwargs,
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = PullRequest(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_pull_request(self, owner: str, repo: str, number: int) -> Response:
        """Get a single pull request."""
        try:
            response = self._session.get(
                url=f"{self.url}/repos/{owner}/{repo}/pulls/{number}",
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = PullRequest(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
