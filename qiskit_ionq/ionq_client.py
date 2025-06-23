# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Copyright 2020 IonQ, Inc. (www.ionq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic API Client for IonQ's REST API"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Optional, TYPE_CHECKING
from warnings import warn
import requests

from . import exceptions
from .helpers import qiskit_to_ionq, get_user_agent, retry
from .exceptions import IonQRetriableError

if TYPE_CHECKING:
    from .ionq_job import IonQJob


class IonQClient:
    """IonQ API Client

    Attributes:
        _url(str): A URL base to use for API calls, e.g. ``"https://api.ionq.co/v0.4"``
        _token(str): An API Access Token to use with the IonQ API.
        _custom_headers(dict): Extra headers to add to the request.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        url: Optional[str] = None,
        custom_headers: Optional[dict] = None,
    ):
        self._token = token
        self._custom_headers = custom_headers or {}
        # strip trailing slashes from our base URL.
        if url and url.endswith("/"):
            url = url[:-1]
        self._url = url
        self._user_agent = get_user_agent()

    @property
    def api_headers(self) -> dict:
        """API Headers needed to make calls to the REST API.

        Returns:
            dict[str, str]: A dict of :class:`requests.Request` headers.
        """
        return {
            **self._custom_headers,
            "Authorization": f"apiKey {self._token}",
            "Content-Type": "application/json",
            "User-Agent": self._user_agent,
        }

    def make_path(self, *parts: str) -> str:
        """Make a "/"-delimited path, then append it to :attr:`_url`.

        Returns:
            str: A URL to use for an API call.
        """
        return f"{self._url}/{'/'.join(parts)}"

    def get_with_retry(self, req_path, params=None, headers=None, timeout=30):
        """Make a GET request with retry logic and exception handling.

        Args:
            req_path (str): The URL path to make the request to.
            params (dict, optional): Parameters to include in the request.
            headers (dict, optional): Headers to include in the request.
            timeout (int, optional): Timeout for the request.

        Raises:
            IonQRetriableError: When a retriable error occurs during the request.

        Returns:
            Response: A requests.Response object.
        """
        try:
            res = requests.get(
                req_path,
                params=params,
                headers=headers,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as req_exc:
            raise IonQRetriableError(req_exc) from req_exc

        return res

    @retry(exceptions=IonQRetriableError, tries=5)
    def submit_job(self, job: IonQJob) -> dict:
        """Submit job to IonQ API

        This returns a JSON dict with status "submitted" and the job's id.

        Args:
            job (IonQJob): The IonQ Job instance to submit to the API.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """
        as_json = qiskit_to_ionq(
            job.circuit,
            job.backend(),
            job._passed_args,
            job.extra_query_params,
            job.extra_metadata,
        )
        req_path = self.make_path("jobs")
        print(
            f"{req_path=}",
            f"as_json={json.dumps(json.loads(as_json), indent=2)}",
            f"{self.api_headers=}",
            sep="\n",
        )
        res = requests.post(
            req_path,
            data=as_json,
            headers=self.api_headers,
            timeout=30,
        )
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    @retry(exceptions=IonQRetriableError, max_delay=60, backoff=2, jitter=1)
    def retrieve_job(self, job_id: str) -> dict:
        """Retrieve job information from the IonQ API.

        The returned JSON dict will only have data if job has completed.

        Args:
            job_id (str): The ID of a job to retrieve.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.
            IonQRetriableError: When a retriable error occurs during the request.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """
        req_path = self.make_path("jobs", job_id)
        res = self.get_with_retry(req_path, headers=self.api_headers)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    @retry(exceptions=IonQRetriableError, tries=5)
    def cancel_job(self, job_id: str) -> dict:
        """Attempt to cancel a job which has not yet run.

        .. NOTE:: If the job has already reached status "completed", this cancel action is a no-op.

        Args:
            job_id (str): The ID of the job to cancel.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """
        req_path = self.make_path("jobs", job_id, "status", "cancel")
        res = requests.put(req_path, headers=self.api_headers, timeout=30)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    def cancel_jobs(self, job_ids: list[str]) -> list[dict]:
        """Cancel multiple jobs at once.

        Args:
            job_ids (list): A list of job IDs to cancel.

        Returns:
            list: A list of :meth:`cancel_job <cancel_job>` responses.
        """
        return [self.cancel_job(job_id) for job_id in job_ids]

    @retry(exceptions=IonQRetriableError, tries=3)
    def delete_job(self, job_id: str) -> dict:
        """Delete a job and associated data.

        Args:
            job_id (str): The ID of the job to delete.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """
        req_path = self.make_path("jobs", job_id)
        res = requests.delete(req_path, headers=self.api_headers, timeout=30)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    @retry(exceptions=IonQRetriableError, max_delay=60, backoff=2, jitter=1)
    def get_characterizations(
        self, backend_name: str, *, limit: int | None = None
    ) -> list[dict]:
        """List all characterizations for a backend."""
        params = {"limit": limit} if limit else None
        url = self.make_path("backends", backend_name, "characterizations")
        res = self.get_with_retry(url, headers=self.api_headers, params=params)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json().get("characterizations", [])

    def get_characterization(self, backend_name: str, char_id: str) -> dict:
        """Fetch a specific characterization by UUID."""
        url = self.make_path("backends", backend_name, "characterizations", char_id)
        res = self.get_with_retry(url, headers=self.api_headers)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    @retry(exceptions=IonQRetriableError, max_delay=60, backoff=2, jitter=1)
    def get_results(
        self,
        job_id: str,
        sharpen: Optional[bool] = None,
        extra_query_params: Optional[dict] = None,
    ) -> dict:
        """Retrieve job results from the IonQ API.

        The returned JSON dict will only have data if job has completed.

        Args:
            job_id (str): The ID of a job to retrieve.
            sharpen (bool): Supported if the job is debiased,
            allows you to filter out physical qubit bias from the results.
            extra_query_params (dict): Specify any parameters to include in the request

        Raises:
            IonQAPIError: When the API returns a non-200 status code.
            IonQRetriableError: When a retriable error occurs during the request.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """

        params = {}

        if sharpen is not None:
            params["sharpen"] = sharpen

        if extra_query_params is not None:
            warn(
                (
                    f"The parameter(s): {extra_query_params} is not checked by default "
                    "but will be submitted in the request."
                )
            )
            params.update(extra_query_params)

        req_path = self.make_path("jobs", job_id, "results", "histogram")
        res = self.get_with_retry(req_path, headers=self.api_headers, params=params)
        exceptions.IonQAPIError.raise_for_status(res)
        # Use json.loads with object_pairs_hook to maintain order of JSON keys
        return json.loads(res.text, object_pairs_hook=OrderedDict)

    def estimate_job(
        self,
        *,
        backend: str,
        oneq_gates: int,
        twoq_gates: int,
        qubits: int,
        shots: int,
        error_mitigation: bool = False,
        session: bool = False,
        job_type: str = "ionq.circuit.v1",
    ) -> dict:
        """Call GET /jobs/estimate … returns a cost/time prediction."""
        params = {
            "type": job_type,
            "backend": backend,
            "1q_gates": oneq_gates,
            "2q_gates": twoq_gates,
            "qubits": qubits,
            "shots": shots,
            "error_mitigation": str(error_mitigation).lower(),
            "session": str(session).lower(),
        }
        url = self.make_path("jobs", "estimate")
        res = self.get_with_retry(url, headers=self.api_headers, params=params)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    @retry(exceptions=IonQRetriableError, tries=5)
    def post(self, *path_parts: str, json_body: dict | None = None) -> dict:
        """POST helper with IonQ headers + retry.

        Args:
            *path_parts (str): Path parts to append to the base URL.
            json_body (dict, optional): JSON body to send in the POST request.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.
            IonQRetriableError: When a retriable error occurs during the request.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """
        url = self.make_path(*path_parts)
        res = requests.post(url, json=json_body, headers=self.api_headers, timeout=30)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    @retry(exceptions=IonQRetriableError, tries=3)
    def put(self, *path_parts: str, json_body: dict | None = None) -> dict:
        """PUT helper with IonQ headers + retry.

        Args:
            *path_parts (str): Path parts to append to the base URL.
            json_body (dict, optional): JSON body to send in the PUT request.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.
            IonQRetriableError: When a retriable error occurs during the request.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """
        url = self.make_path(*path_parts)
        res = requests.put(url, json=json_body, headers=self.api_headers, timeout=30)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()


__all__ = ["IonQClient"]
