#!/usr/bin/env python
from agent_utilities.core.decorators import require_auth

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_response_models import Response


class Api(BaseApiClient):
    # --- Issue / pull-request comments -----------------------------------
    # A pull request IS an issue on GitHub, so these endpoints work for both.

    @require_auth
    def list_issue_comments(
        self, owner: str, repo: str, issue_number: int, **filters
    ) -> Response:
        """List comments on an issue or pull request.

        GET /repos/{owner}/{repo}/issues/{issue_number}/comments. Optional
        filters passed straight through as query parameters (since, per_page,
        page). A pull request IS an issue on GitHub, so this also lists a
        PR's top-level (non-review) comments. Returns the raw JSON list of
        comments in ``Response.data``.
        """
        params = {k: v for k, v in filters.items() if v is not None}
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/issues/{issue_number}/comments",
            params=params or None,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def list_repo_issue_comments(self, owner: str, repo: str, **filters) -> Response:
        """List every issue/PR comment in a repository.

        GET /repos/{owner}/{repo}/issues/comments. Optional filters passed
        straight through as query parameters (sort, direction, since,
        per_page, page). Returns the raw JSON list of comments in
        ``Response.data``.
        """
        params = {k: v for k, v in filters.items() if v is not None}
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/issues/comments",
            params=params or None,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def get_issue_comment(self, owner: str, repo: str, comment_id: int) -> Response:
        """Get a single issue/PR comment.

        GET /repos/{owner}/{repo}/issues/comments/{comment_id}.
        """
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/issues/comments/{comment_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def create_issue_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> Response:
        """Create a comment on an issue or pull request.

        POST /repos/{owner}/{repo}/issues/{issue_number}/comments. A pull
        request IS an issue on GitHub, so this is also how you reply to a PR
        (a top-level comment — for an inline code review comment see
        create_review_comment).
        """
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/issues/{issue_number}/comments",
            json={"body": body},
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def update_issue_comment(
        self, owner: str, repo: str, comment_id: int, body: str
    ) -> Response:
        """Edit an issue/PR comment.

        PATCH /repos/{owner}/{repo}/issues/comments/{comment_id}.
        """
        response = self._session.patch(
            url=f"{self.url}/repos/{owner}/{repo}/issues/comments/{comment_id}",
            json={"body": body},
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def delete_issue_comment(self, owner: str, repo: str, comment_id: int) -> Response:
        """Permanently delete an issue/PR comment.

        DELETE /repos/{owner}/{repo}/issues/comments/{comment_id} (HTTP 204).
        """
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/issues/comments/{comment_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deleted"})

    # --- Pull-request review comments (inline code comments) -------------

    @require_auth
    def list_review_comments(
        self, owner: str, repo: str, pull_number: int, **filters
    ) -> Response:
        """List review (inline code) comments on a pull request.

        GET /repos/{owner}/{repo}/pulls/{pull_number}/comments. Optional
        filters passed straight through as query parameters (since, per_page,
        page). Returns the raw JSON list of comments in ``Response.data``.
        """
        params = {k: v for k, v in filters.items() if v is not None}
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/{pull_number}/comments",
            params=params or None,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def list_repo_review_comments(self, owner: str, repo: str, **filters) -> Response:
        """List every pull-request review comment in a repository.

        GET /repos/{owner}/{repo}/pulls/comments. Optional filters passed
        straight through as query parameters (sort, direction, since,
        per_page, page). Returns the raw JSON list of comments in
        ``Response.data``.
        """
        params = {k: v for k, v in filters.items() if v is not None}
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/comments",
            params=params or None,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def get_review_comment(self, owner: str, repo: str, comment_id: int) -> Response:
        """Get a single pull-request review comment.

        GET /repos/{owner}/{repo}/pulls/comments/{comment_id}.
        """
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/comments/{comment_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def create_review_comment(
        self, owner: str, repo: str, pull_number: int, body: str, **kwargs
    ) -> Response:
        """Create an inline review comment on a pull request's diff.

        POST /repos/{owner}/{repo}/pulls/{pull_number}/comments. Requires
        either a new-thread payload (``commit_id``, ``path``, plus ``line`` +
        ``side``, or ``start_line`` + ``start_side`` for a multi-line
        comment, or the legacy ``position``) or ``in_reply_to`` (an existing
        review comment id) to reply within its thread — the simpler
        create_review_comment_reply wraps that second case.
        """
        payload = {"body": body, **kwargs}
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/{pull_number}/comments",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def create_review_comment_reply(
        self, owner: str, repo: str, pull_number: int, comment_id: int, body: str
    ) -> Response:
        """Reply to an existing pull-request review comment thread.

        POST /repos/{owner}/{repo}/pulls/{pull_number}/comments/{comment_id}/replies.
        Simpler than create_review_comment(in_reply_to=...) — no commit_id or
        path required.
        """
        response = self._session.post(
            url=(
                f"{self.url}/repos/{owner}/{repo}/pulls/{pull_number}/comments/"
                f"{comment_id}/replies"
            ),
            json={"body": body},
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def update_review_comment(
        self, owner: str, repo: str, comment_id: int, body: str
    ) -> Response:
        """Edit a pull-request review comment.

        PATCH /repos/{owner}/{repo}/pulls/comments/{comment_id}.
        """
        response = self._session.patch(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/comments/{comment_id}",
            json={"body": body},
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def delete_review_comment(self, owner: str, repo: str, comment_id: int) -> Response:
        """Permanently delete a pull-request review comment.

        DELETE /repos/{owner}/{repo}/pulls/comments/{comment_id} (HTTP 204).
        """
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/pulls/comments/{comment_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deleted"})

    # --- Commit comments ---------------------------------------------------

    @require_auth
    def list_commit_comments(
        self, owner: str, repo: str, sha: str, **filters
    ) -> Response:
        """List comments on a single commit.

        GET /repos/{owner}/{repo}/commits/{sha}/comments. Optional filters
        passed straight through as query parameters (per_page, page).
        Returns the raw JSON list of comments in ``Response.data``.
        """
        params = {k: v for k, v in filters.items() if v is not None}
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/commits/{sha}/comments",
            params=params or None,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def list_repo_commit_comments(self, owner: str, repo: str, **filters) -> Response:
        """List every commit comment in a repository.

        GET /repos/{owner}/{repo}/comments. Optional filters passed straight
        through as query parameters (per_page, page). Returns the raw JSON
        list of comments in ``Response.data``.
        """
        params = {k: v for k, v in filters.items() if v is not None}
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/comments",
            params=params or None,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def get_commit_comment(self, owner: str, repo: str, comment_id: int) -> Response:
        """Get a single commit comment.

        GET /repos/{owner}/{repo}/comments/{comment_id}.
        """
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/comments/{comment_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def create_commit_comment(
        self, owner: str, repo: str, sha: str, body: str, **kwargs
    ) -> Response:
        """Create a comment on a commit.

        POST /repos/{owner}/{repo}/commits/{sha}/comments. Optionally anchor
        the comment to a line of a file in the commit's diff via ``path``
        (the file path) and ``line`` (the line number in the file).
        """
        payload = {"body": body, **kwargs}
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/commits/{sha}/comments",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def update_commit_comment(
        self, owner: str, repo: str, comment_id: int, body: str
    ) -> Response:
        """Edit a commit comment.

        PATCH /repos/{owner}/{repo}/comments/{comment_id}.
        """
        response = self._session.patch(
            url=f"{self.url}/repos/{owner}/{repo}/comments/{comment_id}",
            json={"body": body},
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def delete_commit_comment(self, owner: str, repo: str, comment_id: int) -> Response:
        """Permanently delete a commit comment.

        DELETE /repos/{owner}/{repo}/comments/{comment_id} (HTTP 204).
        """
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/comments/{comment_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deleted"})
