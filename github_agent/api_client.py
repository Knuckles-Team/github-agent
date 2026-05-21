#!/usr/bin/python

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

import requests
import urllib3
from agent_utilities.base_utilities import get_logger
from pydantic import BaseModel, ValidationError

logger = get_logger(__name__)

from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
    AuthError,
    MissingParameterError,
    ParameterError,
    UnauthorizedError,
)

from github_agent.github_input_models import (
    BranchModel,
    CommitModel,
    ContentModel,
    IssueModel,
    PullRequestModel,
    RepoModel,
)
from github_agent.github_response_models import (
    Branch,
    Commit,
    Content,
    Issue,
    PullRequest,
    Repository,
    Response,
)

T = TypeVar("T", bound=BaseModel)


class Api:
    def __init__(
        self,
        url: str = "https://api.github.com",
        token: str | None = None,
        proxies: dict | None = None,
        verify: bool = True,
        debug: bool = False,
    ):
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.ERROR)

        if url is None:
            raise MissingParameterError

        self._session = requests.Session()
        self.url = url.rstrip("/")
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.verify = verify
        self.proxies = proxies
        self.debug = debug

        if self.verify is False:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        else:
            logger.warning("No token provided for GitHub API")

        try:
            response = self._session.get(
                url=f"{self.url}/user",
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            if response.status_code in (401, 403):
                logger.error(f"Authentication Error: {response.text}")
                raise AuthError if response.status_code == 401 else UnauthorizedError
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection Error: {str(e)}")

    def _fetch_next_page(
        self, endpoint: str, model: T, header: dict, page: int
    ) -> list[dict]:
        """Fetch a single page of data from the specified endpoint"""
        model.page = page  # type: ignore[attr-defined]
        model.model_post_init(None)
        response = self._session.get(
            url=f"{self.url}{endpoint}" if endpoint.startswith("/") else endpoint,
            params=model.api_parameters,  # type: ignore[attr-defined]
            headers=header,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        page_data = response.json()
        return page_data if isinstance(page_data, list) else []

    def _get_total_pages(self, response: requests.Response) -> int:
        """Extract total pages from GitHub Link header"""
        link = response.headers.get("Link")
        if not link:
            return 1

        last_match = re.search(r'page=(\d+)>; rel="last"', link)
        if last_match:
            return int(last_match.group(1))
        return 1

    def _fetch_all_pages(
        self, endpoint: str, model: T
    ) -> tuple[requests.Response, list[dict]]:
        """Generic method to fetch all pages with parallelization if possible"""
        all_data = []

        initial_url = f"{self.url}{endpoint}" if endpoint.startswith("/") else endpoint

        response = self._session.get(
            url=initial_url,
            params=model.api_parameters,  # type: ignore[attr-defined]
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        initial_data = response.json()

        if isinstance(initial_data, list):
            all_data.extend(initial_data)
        else:
            return response, [initial_data]

        total_pages = self._get_total_pages(response)

        max_pages = getattr(model, "max_pages", total_pages)
        if not max_pages or max_pages == 0 or max_pages > total_pages:
            max_pages = total_pages
            model.max_pages = total_pages  # type: ignore[attr-defined]

        if max_pages > 1:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for page in range(2, max_pages + 1):
                    futures.append(
                        executor.submit(
                            self._fetch_next_page,
                            initial_url,
                            model,
                            self.headers,
                            page,
                        )
                    )

                for future in as_completed(futures):
                    try:
                        all_data.extend(future.result())
                    except Exception as e:
                        logger.error(f"Error fetching page: {str(e)}")

        return response, all_data

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
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Repository(**response.json())
            return Response(response=response, data=parsed_data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ParameterError(f"Repository {owner}/{repo} not found") from e
            raise

    @require_auth
    def get_issues(self, **kwargs) -> Response:
        """List issues for a repository."""
        model = IssueModel(**kwargs)
        if not model.owner or not model.repo:
            raise MissingParameterError("owner and repo are required")
        try:
            response, data = self._fetch_all_pages(
                f"/repos/{model.owner}/{model.repo}/issues", model
            )
            parsed_data = [Issue(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_pull_requests(self, **kwargs) -> Response:
        """List pull requests for a repository."""
        model = PullRequestModel(**kwargs)
        if not model.owner or not model.repo:
            raise MissingParameterError("owner and repo are required")
        try:
            response, data = self._fetch_all_pages(
                f"/repos/{model.owner}/{model.repo}/pulls", model
            )
            parsed_data = [PullRequest(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_contents(self, **kwargs) -> Response:
        """Get contents of a file or directory in a repository."""
        model = ContentModel(**kwargs)
        try:
            response = self._session.get(
                url=f"{self.url}/repos/{model.owner}/{model.repo}/contents/{model.path}",
                params=model.api_parameters,
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
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
    def create_repository(self, name: str, **kwargs) -> Response:
        """Create a new repository for the authenticated user."""
        try:
            payload = {"name": name, **kwargs}
            response = self._session.post(
                url=f"{self.url}/user/repos",
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
    def delete_repository(self, owner: str, repo: str) -> Response:
        """Delete a repository."""
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
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
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Repository(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def create_issue(self, owner: str, repo: str, title: str, **kwargs) -> Response:
        """Create a new issue in a repository."""
        try:
            payload = {"title": title, **kwargs}
            response = self._session.post(
                url=f"{self.url}/repos/{owner}/{repo}/issues",
                json=payload,
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Issue(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def update_issue(self, owner: str, repo: str, number: int, **kwargs) -> Response:
        """Update an issue in a repository."""
        try:
            response = self._session.patch(
                url=f"{self.url}/repos/{owner}/{repo}/issues/{number}",
                json=kwargs,
                headers=self.headers,
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Issue(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_issue(self, owner: str, repo: str, number: int) -> Response:
        """Get a single issue in a repository."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/issues/{number}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        parsed_data = Issue(**response.json())
        return Response(response=response, data=parsed_data)

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
                verify=self.verify,
                proxies=self.proxies,
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
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = PullRequest(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_pull_request(self, owner: str, repo: str, number: int) -> Response:
        """Get a single pull request."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/{number}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        parsed_data = PullRequest(**response.json())
        return Response(response=response, data=parsed_data)

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
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Content(**response.json()["content"])
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
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            parsed_data = Content(**response.json()["content"])
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
                verify=self.verify,
                proxies=self.proxies,
            )
            response.raise_for_status()
            return Response(response=response, data=response.json())
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
                verify=self.verify,
                proxies=self.proxies,
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
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deleted"})

    @require_auth
    def get_branch(self, owner: str, repo: str, branch: str) -> Response:
        """Get a single branch in a repository."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/branches/{branch}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        parsed_data = Branch(**response.json())
        return Response(response=response, data=parsed_data)

    @require_auth
    def get_commit(self, owner: str, repo: str, sha: str) -> Response:
        """Get a single commit in a repository."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/commits/{sha}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        parsed_data = Commit(**response.json())
        return Response(response=response, data=parsed_data)
