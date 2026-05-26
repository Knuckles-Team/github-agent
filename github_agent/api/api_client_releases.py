#!/usr/bin/env python
from agent_utilities.decorators import require_auth
from agent_utilities.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_response_models import (
    Release,
    Response,
)


class Api(BaseApiClient):
    @require_auth
    def get_releases(self, owner: str, repo: str) -> Response:
        """List repository releases."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/releases",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            res_json = response.json()
            if not isinstance(res_json, list):
                raise ParameterError("Invalid parameters: expected list of releases")
            parsed_data = [Release(**item) for item in res_json]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_release(self, owner: str, repo: str, release_id: int) -> Response:
        """Get a single repository release."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/releases/{release_id}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = Release(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_release(
        self,
        owner: str,
        repo: str,
        tag_name: str,
        name: str | None = None,
        body: str | None = None,
        draft: bool = False,
        prerelease: bool = False,
    ) -> Response:
        """Create a new repository release."""
        payload = {
            "tag_name": tag_name,
            "draft": draft,
            "prerelease": prerelease,
        }
        if name:
            payload["name"] = name
        if body:
            payload["body"] = body
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/releases",
            json=payload,
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = Release(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def update_release(
        self, owner: str, repo: str, release_id: int, **kwargs
    ) -> Response:
        """Update (PATCH) a repository release."""
        response = self._session.patch(
            url=f"{self.url}/repos/{owner}/{repo}/releases/{release_id}",
            json=kwargs,
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = Release(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def delete_release(self, owner: str, repo: str, release_id: int) -> Response:
        """Delete a repository release."""
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/releases/{release_id}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deleted"})
