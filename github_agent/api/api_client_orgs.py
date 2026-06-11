#!/usr/bin/env python
from urllib.parse import urlparse

from agent_utilities.decorators import require_auth
from agent_utilities.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    OrgListModel,
    OrgMemberModel,
    OrgUpdateModel,
)
from github_agent.github_response_models import (
    Organization,
    OrganizationMembership,
    OrganizationSummary,
    Repository,
    Response,
    User,
)


class OrganizationCreationNotSupportedError(Exception):
    """Raised when organization creation is requested against github.com.

    github.com does not expose an API to create organizations — they can only
    be created through the web UI. The POST /admin/organizations endpoint
    exists only on GitHub Enterprise Server.
    """


class Api(BaseApiClient):
    @require_auth
    def get_organization(self, org: str) -> Response:
        """Get an organization's full profile.

        GET /orgs/{org}
        https://docs.github.com/en/rest/orgs/orgs#get-an-organization
        """
        response = self._session.get(
            url=f"{self.url}/orgs/{org}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = Organization(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def list_organizations(self, **kwargs) -> Response:
        """List organizations.

        With scope='member' (default): organizations the authenticated user
        belongs to.
        GET /user/orgs
        https://docs.github.com/en/rest/orgs/orgs#list-organizations-for-the-authenticated-user

        With scope='all': every organization, in ascending ID order. GitHub
        paginates this endpoint with a 'since' ID cursor — pass the highest
        organization ID from the previous page to fetch the next one.
        GET /organizations
        https://docs.github.com/en/rest/orgs/orgs#list-organizations
        """
        model = OrgListModel(**kwargs)
        endpoint = "/user/orgs" if model.scope == "member" else "/organizations"
        response, data = self._fetch_all_pages(endpoint, model)
        try:
            parsed_data = [OrganizationSummary(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def update_organization(self, org: str, **kwargs) -> Response:
        """Update an organization's profile and member settings.

        Accepts the documented mutable fields (validated by OrgUpdateModel):
        billing_email, company, email, twitter_username, location, name,
        description, blog, has_organization_projects, has_repository_projects,
        default_repository_permission, members_can_create_repositories,
        members_can_create_public/private/internal_repositories,
        members_can_create_pages, members_can_fork_private_repositories,
        web_commit_signoff_required.

        PATCH /orgs/{org}
        https://docs.github.com/en/rest/orgs/orgs#update-an-organization
        """
        try:
            model = OrgUpdateModel(**kwargs)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
        response = self._session.patch(
            url=f"{self.url}/orgs/{org}",
            json=model.model_dump(exclude_none=True),
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = Organization(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def delete_organization(self, org: str) -> Response:
        """Schedule an organization for deletion. IRREVERSIBLE.

        GitHub responds 202 Accepted and deletes the organization, all of its
        repositories, and all of its data asynchronously. The MCP layer gates
        this action behind allow_destructive / GITHUB_ALLOW_DESTRUCTIVE.

        DELETE /orgs/{org}
        https://docs.github.com/en/rest/orgs/orgs#delete-an-organization
        """
        response = self._session.delete(
            url=f"{self.url}/orgs/{org}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deletion_scheduled"})

    @require_auth
    def create_organization(
        self, login: str, admin: str, profile_name: str | None = None
    ) -> Response:
        """Create an organization — GitHub Enterprise Server ONLY.

        POST /admin/organizations (site-admin API)
        https://docs.github.com/en/enterprise-server@latest/rest/enterprise-admin/orgs#create-an-organization

        github.com does NOT expose an API to create organizations; on
        github.com they can only be created through the web UI
        (https://github.com/account/organizations/new). When this client
        points at api.github.com this method raises
        OrganizationCreationNotSupportedError instead of calling the API.

        Args:
            login: The organization's username.
            admin: The login of the user who will manage this organization.
            profile_name: The organization's display name.
        """
        host = urlparse(self.url).hostname or ""
        if host == "api.github.com":
            raise OrganizationCreationNotSupportedError(
                "github.com organizations cannot be created via the API — "
                "create them in the web UI at "
                "https://github.com/account/organizations/new. "
                "POST /admin/organizations is only available on GitHub "
                "Enterprise Server (point GITHUB_URL at your enterprise "
                "instance's API, e.g. https://ghe.example.com/api/v3)."
            )
        payload: dict = {"login": login, "admin": admin}
        if profile_name is not None:
            payload["profile_name"] = profile_name
        response = self._session.post(
            url=f"{self.url}/admin/organizations",
            json=payload,
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = OrganizationSummary(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_organization_repository(self, org: str, name: str, **kwargs) -> Response:
        """Create a repository in an organization.

        Accepts the same payload fields as create_repository (which posts to
        /user/repos): description, homepage, private, visibility, has_issues,
        has_projects, has_wiki, auto_init, gitignore_template,
        license_template, etc.

        POST /orgs/{org}/repos
        https://docs.github.com/en/rest/repos/repos#create-an-organization-repository
        """
        try:
            payload = {"name": name, **kwargs}
            response = self._session.post(
                url=f"{self.url}/orgs/{org}/repos",
                json=payload,
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Repository(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_organization_membership(self, org: str, username: str) -> Response:
        """Get a user's organization membership (state and role).

        GET /orgs/{org}/memberships/{username}
        https://docs.github.com/en/rest/orgs/members#get-organization-membership-for-a-user
        """
        response = self._session.get(
            url=f"{self.url}/orgs/{org}/memberships/{username}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = OrganizationMembership(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def set_organization_membership(
        self, org: str, username: str, role: str = "member"
    ) -> Response:
        """Add a user to an organization or update their role.

        Invites the user when they are not yet a member (membership state
        'pending' until accepted). Role is 'member' or 'admin'.

        PUT /orgs/{org}/memberships/{username}
        https://docs.github.com/en/rest/orgs/members#set-organization-membership-for-a-user
        """
        response = self._session.put(
            url=f"{self.url}/orgs/{org}/memberships/{username}",
            json={"role": role},
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = OrganizationMembership(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def remove_organization_member(self, org: str, username: str) -> Response:
        """Remove a user from an organization (repositories access included).

        Destructive: the MCP layer gates this action behind
        allow_destructive / GITHUB_ALLOW_DESTRUCTIVE. GitHub responds 204.

        DELETE /orgs/{org}/members/{username}
        https://docs.github.com/en/rest/orgs/members#remove-an-organization-member
        """
        response = self._session.delete(
            url=f"{self.url}/orgs/{org}/members/{username}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "removed"})

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
