#!/usr/bin/env python
import requests
from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    CollaboratorModel,
    OrgRepoModel,
    RepoModel,
)
from github_agent.github_response_models import (
    Collaborator,
    CollaboratorInvitation,
    Repository,
    Response,
)


class Api(BaseApiClient):
    @require_auth
    def get_repositories(self, **kwargs) -> Response:
        """List repositories for the authenticated user."""
        model = RepoModel(**kwargs)
        try:
            response, data = self._fetch_all_pages("/user/repos", model)
            parsed_data = [Repository(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_repository(self, owner: str, repo: str) -> Response:
        """Get a specific repository."""
        try:
            response = self._session.get(
                url=f"{self.url}/repos/{owner}/{repo}",
                headers=self.headers,
            )
            response.raise_for_status()
            parsed_data = Repository(**response.json())
            return Response(response=response, data=parsed_data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ParameterError(f"Repository {owner}/{repo} not found") from e
            raise

    @require_auth
    def create_repository(self, name: str, **kwargs) -> Response:
        """Create a repository for the authenticated user, or for an organization when 'org' is given.

        Without ``org``: POST /user/repos — the repository is created under
        the authenticated user's personal account.

        With ``org``: POST /orgs/{org}/repos (same payload shape as
        ``create_organization_repository``). ``org`` is consumed as the URL
        path segment and is never forwarded as a JSON body field. This
        distinction matters: GitHub's /user/repos endpoint silently ignores
        an unrecognized "org" body field instead of rejecting it, which
        previously let a caller pass ``org=...`` expecting an organization
        repository and get a 201 Created — with the repository silently
        created under the personal account instead (a wrong-target bug, not
        a clean failure).
        """
        try:
            org = kwargs.pop("org", None)
            payload = {"name": name, **kwargs}
            url = f"{self.url}/orgs/{org}/repos" if org else f"{self.url}/user/repos"
            response = self._session.post(
                url=url,
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            parsed_data = Repository(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def delete_repository(self, owner: str, repo: str) -> Response:
        """Delete a repository."""
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deleted"})

    @require_auth
    def update_repository(self, owner: str, repo: str, **kwargs) -> Response:
        """Update a repository."""
        try:
            response = self._session.patch(
                url=f"{self.url}/repos/{owner}/{repo}",
                json=kwargs,
                headers=self.headers,
            )
            response.raise_for_status()
            parsed_data = Repository(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_org_repos(self, **kwargs) -> Response:
        """List repositories for an organization."""
        model = OrgRepoModel(**kwargs)
        response, data = self._fetch_all_pages(f"/orgs/{model.org}/repos", model)
        try:
            parsed_data = [Repository(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_collaborators(self, **kwargs) -> Response:
        """List collaborators for a repository."""
        model = CollaboratorModel(**kwargs)
        response, data = self._fetch_all_pages(
            f"/repos/{model.owner}/{model.repo}/collaborators", model
        )
        try:
            parsed_data = [Collaborator(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def add_collaborator(
        self, owner: str, repo: str, username: str, permission: str | None = None
    ) -> Response:
        """Add a collaborator to a repository."""
        payload = {}
        if permission:
            payload["permission"] = permission
        try:
            response = self._session.put(
                url=f"{self.url}/repos/{owner}/{repo}/collaborators/{username}",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            if response.status_code == 204:
                return Response(response=response, data={})
            parsed_data = CollaboratorInvitation(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def remove_collaborator(self, owner: str, repo: str, username: str) -> Response:
        """Remove a collaborator from a repository."""
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/collaborators/{username}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "removed"})

    @require_auth
    def get_repo_secret_public_key(self, owner: str, repo: str) -> Response:
        """Get the public key for encrypting repository Actions secrets."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/actions/secrets/public-key",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def get_repo_secrets(self, owner: str, repo: str) -> Response:
        """List repository Actions secrets names."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/actions/secrets",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def create_or_update_repo_secret(
        self, owner: str, repo: str, secret_name: str, encrypted_value: str, key_id: str
    ) -> Response:
        """Create or update a repository Actions secret."""
        payload = {"encrypted_value": encrypted_value, "key_id": key_id}
        response = self._session.put(
            url=f"{self.url}/repos/{owner}/{repo}/actions/secrets/{secret_name}",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(
            response=response, data=response.json() if response.content else {}
        )

    @require_auth
    def delete_repo_secret(self, owner: str, repo: str, secret_name: str) -> Response:
        """Delete a repository Actions secret."""
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/actions/secrets/{secret_name}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "secret_deleted"})
