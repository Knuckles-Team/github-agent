#!/usr/bin/env python
from agent_utilities.decorators import require_auth
from agent_utilities.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    OrgMemberModel,
)
from github_agent.github_response_models import (
    Response,
    User,
)


class Api(BaseApiClient):
    @require_auth
    def get_org_members(self, **kwargs) -> Response:
        """List members for an organization."""
        model = OrgMemberModel(**kwargs)
        response, data = self._fetch_all_pages(f"/orgs/{model.org}/members", model)
        try:
            parsed_data = [User(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_org_teams(self, org: str) -> Response:
        """List teams for an organization."""
        response = self._session.get(
            url=f"{self.url}/orgs/{org}/teams",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())
