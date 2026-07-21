#!/usr/bin/env python
from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    CommitModel,
)
from github_agent.github_response_models import (
    Commit,
    Response,
)


class Api(BaseApiClient):
    @require_auth
    def get_commits(self, **kwargs) -> Response:
        """List commits for a repository."""
        model = CommitModel(**kwargs)
        try:
            response, data = self._fetch_all_pages(
                f"/repos/{model.owner}/{model.repo}/commits", model
            )
            parsed_data = [Commit(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_commit(self, owner: str, repo: str, sha: str) -> Response:
        """Get a single commit in a repository."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/commits/{sha}",
            headers=self.headers,
        )
        response.raise_for_status()
        try:
            parsed_data = Commit(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
