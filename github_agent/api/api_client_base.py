#!/usr/bin/env python
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

import requests
import urllib3
from agent_utilities.base_utilities import get_logger
from agent_utilities.exceptions import (
    AuthError,
    MissingParameterError,
    UnauthorizedError,
)
from pydantic import BaseModel

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class BaseApiClient:
    def __init__(
        self,
        url: str | None = "https://api.github.com",
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
