#!/usr/bin/env python
from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
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
            )
            response.raise_for_status()
            parsed_data = PullRequest(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        number: int,
        event: str = "APPROVE",
        body: str | None = None,
        **kwargs,
    ) -> Response:
        """Create a review on a pull request.

        ``event`` is one of APPROVE, REQUEST_CHANGES, or COMMENT (GitHub requires
        a ``body`` for REQUEST_CHANGES/COMMENT).
        """
        payload = {"event": event, **kwargs}
        if body is not None:
            payload["body"] = body
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/{number}/reviews",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def merge_pull_request(
        self,
        owner: str,
        repo: str,
        number: int,
        merge_method: str = "merge",
        commit_title: str | None = None,
        commit_message: str | None = None,
        sha: str | None = None,
    ) -> Response:
        """Merge a pull request (``merge_method`` is merge, squash, or rebase)."""
        payload: dict = {"merge_method": merge_method}
        if commit_title is not None:
            payload["commit_title"] = commit_title
        if commit_message is not None:
            payload["commit_message"] = commit_message
        if sha is not None:
            payload["sha"] = sha
        response = self._session.put(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/{number}/merge",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def request_reviewers(
        self,
        owner: str,
        repo: str,
        number: int,
        reviewers: list | None = None,
        team_reviewers: list | None = None,
    ) -> Response:
        """Request individual and/or team reviewers on a pull request."""
        payload: dict = {}
        if reviewers is not None:
            payload["reviewers"] = reviewers
        if team_reviewers is not None:
            payload["team_reviewers"] = team_reviewers
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/{number}/requested_reviewers",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        try:
            return Response(response=response, data=PullRequest(**response.json()))
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
