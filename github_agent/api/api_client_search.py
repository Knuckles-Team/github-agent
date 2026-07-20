#!/usr/bin/env python
from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    SearchModel,
)
from github_agent.github_response_models import (
    Response,
    SearchResult,
)


class Api(BaseApiClient):
    @require_auth
    def search_repositories(self, **kwargs) -> Response:
        """Search repositories using query keywords."""
        model = SearchModel(**kwargs)
        response, data = self._fetch_all_pages("/search/repositories", model)
        try:
            parsed_data = (
                SearchResult(**data[0])
                if data
                else SearchResult(total_count=0, incomplete_results=False, items=[])
            )
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def search_issues(self, **kwargs) -> Response:
        """Search issues using query keywords."""
        model = SearchModel(**kwargs)
        response, data = self._fetch_all_pages("/search/issues", model)
        try:
            parsed_data = (
                SearchResult(**data[0])
                if data
                else SearchResult(total_count=0, incomplete_results=False, items=[])
            )
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def search_code(self, **kwargs) -> Response:
        """Search code using query keywords."""
        model = SearchModel(**kwargs)
        response, data = self._fetch_all_pages("/search/code", model)
        try:
            parsed_data = (
                SearchResult(**data[0])
                if data
                else SearchResult(total_count=0, incomplete_results=False, items=[])
            )
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
