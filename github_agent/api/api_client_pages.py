#!/usr/bin/env python
import requests
from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    PagesBuildsModel,
    PagesUpdateModel,
)
from github_agent.github_response_models import (
    PagesAlreadyEnabled,
    PagesBuild,
    PagesBuildRequest,
    PagesNotEnabled,
    PagesSite,
    Response,
)


def _normalize_pages_source(source: str | dict) -> dict:
    """Coerce a Pages publishing source into the API's {'branch', 'path'} shape.

    Accepts a branch name string or a {'branch': ..., 'path': ...} mapping;
    path defaults to '/'.
    """
    if isinstance(source, str):
        return {"branch": source, "path": "/"}
    if isinstance(source, dict) and "branch" in source:
        return {"branch": source["branch"], "path": source.get("path", "/")}
    raise ParameterError(
        "Invalid 'source': expected a branch name string or a mapping with "
        "'branch' (and an optional 'path' of '/' or '/docs')."
    )


class Api(BaseApiClient):
    @require_auth
    def get_pages(self, owner: str, repo: str) -> Response:
        """Get the GitHub Pages site configuration for a repository.

        Returns PagesSite data when Pages is enabled. When the repository has
        no Pages site GitHub responds 404 — instead of raising, this returns
        a typed PagesNotEnabled result so callers can branch on it.

        GET /repos/{owner}/{repo}/pages
        https://docs.github.com/en/rest/pages/pages#get-a-apiname-pages-site
        """
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/pages",
            headers=self.headers,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return Response(response=response, data=PagesNotEnabled())
            raise
        try:
            parsed_data = PagesSite(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_pages(
        self,
        owner: str,
        repo: str,
        build_type: str = "workflow",
        source: str | dict | None = None,
    ) -> Response:
        """Enable GitHub Pages for a repository (HTTP 201).

        build_type 'workflow' (default) publishes via GitHub Actions and
        needs no source; 'legacy' publishes from a branch and requires
        source (a branch name string or {'branch', 'path'} mapping).

        When Pages is already enabled GitHub responds 409 Conflict — instead
        of raising, this returns a typed PagesAlreadyEnabled result.

        POST /repos/{owner}/{repo}/pages
        https://docs.github.com/en/rest/pages/pages#create-a-apiname-pages-site
        """
        if build_type not in ("legacy", "workflow"):
            raise ParameterError(
                "Invalid 'build_type': must be 'legacy' or 'workflow'."
            )
        payload: dict = {"build_type": build_type}
        if source is not None:
            payload["source"] = _normalize_pages_source(source)
        elif build_type == "legacy":
            raise ParameterError(
                "build_type 'legacy' requires 'source' (a branch name string "
                "or a {'branch', 'path'} mapping)."
            )
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/pages",
            json=payload,
            headers=self.headers,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                return Response(response=response, data=PagesAlreadyEnabled())
            raise
        try:
            parsed_data = PagesSite(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def update_pages(self, owner: str, repo: str, **kwargs) -> Response:
        """Update the GitHub Pages configuration for a repository.

        Accepts the documented mutable fields (validated by PagesUpdateModel):
        build_type ('legacy' or 'workflow'), source (a branch name string or
        {'branch', 'path'} mapping, for legacy builds), cname (custom
        domain), https_enforced. GitHub responds 204 No Content.

        PUT /repos/{owner}/{repo}/pages
        https://docs.github.com/en/rest/pages/pages#update-information-about-a-apiname-pages-site
        """
        try:
            model = PagesUpdateModel(**kwargs)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
        payload = model.model_dump(exclude_none=True)
        if not payload:
            raise ParameterError(
                "No fields to update: pass at least one of build_type, "
                "source, cname, https_enforced."
            )
        if "source" in payload:
            payload["source"] = _normalize_pages_source(payload["source"])
        response = self._session.put(
            url=f"{self.url}/repos/{owner}/{repo}/pages",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "pages_updated"})

    @require_auth
    def delete_pages(self, owner: str, repo: str) -> Response:
        """Disable GitHub Pages for a repository and delete the site.

        Destructive: the MCP layer gates this action behind
        allow_destructive / GITHUB_ALLOW_DESTRUCTIVE. GitHub responds 204.

        DELETE /repos/{owner}/{repo}/pages
        https://docs.github.com/en/rest/pages/pages#delete-a-apiname-pages-site
        """
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/pages",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "pages_deleted"})

    @require_auth
    def get_pages_build_latest(self, owner: str, repo: str) -> Response:
        """Get the latest GitHub Pages build for a repository.

        GET /repos/{owner}/{repo}/pages/builds/latest
        https://docs.github.com/en/rest/pages/pages#get-latest-pages-build
        """
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/pages/builds/latest",
            headers=self.headers,
        )
        response.raise_for_status()
        try:
            parsed_data = PagesBuild(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def list_pages_builds(self, **kwargs) -> Response:
        """List GitHub Pages builds for a repository (newest first).

        GET /repos/{owner}/{repo}/pages/builds
        https://docs.github.com/en/rest/pages/pages#list-apiname-pages-builds
        """
        model = PagesBuildsModel(**kwargs)
        response, data = self._fetch_all_pages(
            f"/repos/{model.owner}/{model.repo}/pages/builds", model
        )
        try:
            parsed_data = [PagesBuild(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def request_pages_build(self, owner: str, repo: str) -> Response:
        """Request a fresh GitHub Pages build without pushing a commit.

        The programmatic fix for the first-deploy race where the initial
        Pages build never runs (e.g. Pages enabled after the deploy workflow
        already finished): trigger a build of the latest revision on demand.
        GitHub responds 201 with the queued build request.

        POST /repos/{owner}/{repo}/pages/builds
        https://docs.github.com/en/rest/pages/pages#request-a-apiname-pages-build
        """
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/pages/builds",
            headers=self.headers,
        )
        response.raise_for_status()
        try:
            parsed_data = PagesBuildRequest(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e
