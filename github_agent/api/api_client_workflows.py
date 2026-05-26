#!/usr/bin/env python
from typing import Any

from agent_utilities.decorators import require_auth
from agent_utilities.exceptions import (
    ParameterError,
)
from pydantic import ValidationError

from github_agent.api.api_client_base import BaseApiClient
from github_agent.github_input_models import (
    WorkflowRunModel,
)
from github_agent.github_response_models import (
    Response,
    Workflow,
    WorkflowRun,
)


class Api(BaseApiClient):
    @require_auth
    def get_workflows(self, owner: str, repo: str) -> Response:
        """List workflows for a repository."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/actions/workflows",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            res_json = response.json()
            if not isinstance(res_json, dict) or "workflows" not in res_json:
                raise ParameterError("Invalid parameters: missing 'workflows' key")
            data = res_json.get("workflows", [])
            parsed_data = [Workflow(**item) for item in data]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_workflow_runs(self, **kwargs) -> Response:
        """List workflow runs for a repository."""
        model = WorkflowRunModel(**kwargs)
        response, data = self._fetch_all_pages(
            f"/repos/{model.owner}/{model.repo}/actions/runs", model
        )
        try:
            runs = []
            for item in data:
                if isinstance(item, dict) and "workflow_runs" in item:
                    runs.extend(item["workflow_runs"])
                else:
                    runs.append(item)
            parsed_data = [WorkflowRun(**run) for run in runs]
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def get_workflow_run(self, owner: str, repo: str, run_id: int) -> Response:
        """Get a single workflow run."""
        response = self._session.get(
            url=f"{self.url}/repos/{owner}/{repo}/actions/runs/{run_id}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        try:
            parsed_data = WorkflowRun(**response.json())
            return Response(response=response, data=parsed_data)
        except ValidationError as e:
            raise ParameterError(f"Invalid parameters: {e.errors()}") from e

    @require_auth
    def trigger_workflow_dispatch(
        self,
        owner: str,
        repo: str,
        workflow_id: str | int,
        ref: str,
        inputs: dict | None = None,
    ) -> Response:
        """Trigger a workflow dispatch event."""
        payload: dict[str, Any] = {"ref": ref}
        if inputs:
            payload["inputs"] = inputs
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
            json=payload,
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "dispatched"})

    @require_auth
    def rerun_workflow_run(self, owner: str, repo: str, run_id: int) -> Response:
        """Re-run a workflow run."""
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/actions/runs/{run_id}/rerun",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "rerun_triggered"})

    @require_auth
    def cancel_workflow_run(self, owner: str, repo: str, run_id: int) -> Response:
        """Cancel a workflow run."""
        response = self._session.post(
            url=f"{self.url}/repos/{owner}/{repo}/actions/runs/{run_id}/cancel",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "cancelled"})

    @require_auth
    def delete_workflow_run(self, owner: str, repo: str, run_id: int) -> Response:
        """Delete a workflow run."""
        response = self._session.delete(
            url=f"{self.url}/repos/{owner}/{repo}/actions/runs/{run_id}",
            headers=self.headers,
            verify=self.verify,
            proxies=self.proxies,
        )
        response.raise_for_status()
        return Response(response=response, data={"status": "deleted"})
