#!/usr/bin/env python
from typing import Any

from agent_utilities.decorators import require_auth

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_response_models import Response


class Api(BaseApiClient):
    @require_auth
    def get_dependabot_alerts(self, owner: str, repo: str, **filters) -> Response:
        """List Dependabot alerts for a repository.

        GET /repos/{owner}/{repo}/dependabot/alerts. Optional filters passed
        straight through as query parameters (state, severity, ecosystem,
        package, scope, sort, direction, per_page, page). Returns the raw JSON
        list of alerts in ``Response.data``.
        """
        params = {k: v for k, v in filters.items() if v is not None}
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/dependabot/alerts",
            params=params or None,
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def get_dependabot_alert(
        self, owner: str, repo: str, alert_number: int
    ) -> Response:
        """Get a single Dependabot alert.

        GET /repos/{owner}/{repo}/dependabot/alerts/{alert_number}.
        """
        response = self._session.get(
            url=(f"{self.url}/repos/{owner}/{repo}/dependabot/alerts/{alert_number}"),
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def get_org_dependabot_alerts(self, org: str, **filters) -> Response:
        """List Dependabot alerts for an entire organization.

        GET /orgs/{org}/dependabot/alerts. Optional filters passed straight
        through as query parameters (state, severity, ecosystem, package,
        scope, sort, direction, per_page, page). Returns the raw JSON list of
        alerts in ``Response.data``.
        """
        params = {k: v for k, v in filters.items() if v is not None}
        response = self._session.get(
            url=f"{self.url}/orgs/{org}/dependabot/alerts",
            params=params or None,
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())

    @require_auth
    def update_dependabot_alert(
        self,
        owner: str,
        repo: str,
        alert_number: int,
        state: str,
        dismissed_reason: str | None = None,
        dismissed_comment: str | None = None,
    ) -> Response:
        """Update the state of a Dependabot alert.

        PATCH /repos/{owner}/{repo}/dependabot/alerts/{alert_number}. ``state``
        is ``dismissed`` or ``open``. ``dismissed_reason`` is required when
        dismissing (one of ``fix_started``, ``inaccurate``, ``no_bandwidth``,
        ``not_used``, ``tolerable_risk``); ``dismissed_comment`` is optional.
        """
        payload: dict[str, Any] = {"state": state}
        if dismissed_reason is not None:
            payload["dismissed_reason"] = dismissed_reason
        if dismissed_comment is not None:
            payload["dismissed_comment"] = dismissed_comment
        response = self._session.patch(
            url=(f"{self.url}/repos/{owner}/{repo}/dependabot/alerts/{alert_number}"),
            json=payload,
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data=response.json())
