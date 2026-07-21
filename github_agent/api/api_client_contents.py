#!/usr/bin/env python
from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    ContentModel,
)
from github_agent.github_response_models import (
    Content,
    Response,
)


class Api(BaseApiClient):
    @require_auth
    def get_contents(self, **kwargs) -> Response:
        """Get contents of a file or directory in a repository."""
        model = ContentModel(**kwargs)
        try:
            response = self._session.get(
                url=f"{self.url}/repos/{model.owner}/{model.repo}/contents/{model.path}",
                params=model.api_parameters,
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
            parsed_data: list[Content] | Content
            if isinstance(data, list):
                parsed_data = [Content(**item) for item in data]
            else:
                parsed_data = Content(**data)
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_content(
        self, owner: str, repo: str, path: str, message: str, content: str, **kwargs
    ) -> Response:
        """Create a file in a repository."""
        try:
            payload = {"message": message, "content": content, **kwargs}
            response = self._session.put(
                url=f"{self.url}/repos/{owner}/{repo}/contents/{path}",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            res_json = response.json()
            if not isinstance(res_json, dict) or "content" not in res_json:
                raise ParameterError("Invalid parameters: missing 'content' key")
            parsed_data = Content(**res_json["content"])
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def update_content(
        self,
        owner: str,
        repo: str,
        path: str,
        message: str,
        content: str,
        sha: str,
        **kwargs,
    ) -> Response:
        """Update a file in a repository."""
        try:
            payload = {"message": message, "content": content, "sha": sha, **kwargs}
            response = self._session.put(
                url=f"{self.url}/repos/{owner}/{repo}/contents/{path}",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            res_json = response.json()
            if not isinstance(res_json, dict) or "content" not in res_json:
                raise ParameterError("Invalid parameters: missing 'content' key")
            parsed_data = Content(**res_json["content"])
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def delete_content(
        self, owner: str, repo: str, path: str, message: str, sha: str, **kwargs
    ) -> Response:
        """Delete a file in a repository."""
        try:
            payload = {"message": message, "sha": sha, **kwargs}
            response = self._session.delete(
                url=f"{self.url}/repos/{owner}/{repo}/contents/{path}",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            return Response(response=response, data=response.json())
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
