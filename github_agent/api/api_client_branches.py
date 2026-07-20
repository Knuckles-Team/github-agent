#!/usr/bin/env python
from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    BranchModel,
)
from github_agent.github_response_models import (
    Branch,
    Response,
)


class Api(BaseApiClient):
    @require_auth
    def get_branches(self, **kwargs) -> Response:
        """List branches for a repository."""
        model = BranchModel(**kwargs)
        try:
            response, data = self._fetch_all_pages(
                f"/repos/{model.owner}/{model.repo}/branches", model
            )
            parsed_data = [Branch(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_branch(self, owner: str, repo: str, branch: str, ref: str) -> Response:
        """Create a new branch in a repository (using git ref creation)."""
        try:
            payload = {"ref": f"refs/heads/{branch}", "sha": ref}
            response = self._session.post(
                url=f"{self.url}/repos/{owner}/{repo}/git/refs",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            return Response(response=response, data=response.json())
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def delete_branch(self, owner: str, repo: str, branch: str) -> Response:
        """Delete a branch in a repository."""
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/git/refs/heads/{branch}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deleted"})

    @require_auth
    def get_branch(self, owner: str, repo: str, branch: str) -> Response:
        """Get a single branch in a repository."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/branches/{branch}",
            headers=self.headers,
        )
        response.raise_for_status()
        try:
            parsed_data = Branch(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_branch_protection(self, owner: str, repo: str, branch: str) -> Response:
        """Get branch protection configuration."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/branches/{branch}/protection",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def update_branch_protection(
        self, owner: str, repo: str, branch: str, protection_config: dict
    ) -> Response:
        """Update branch protection configuration."""
        response = self._session.put(
            url=f"{self.url}/repos/{owner}/{repo}/branches/{branch}/protection",
            json=protection_config,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def delete_branch_protection(self, owner: str, repo: str, branch: str) -> Response:
        """Delete branch protection configuration."""
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/branches/{branch}/protection",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "protection_deleted"})
