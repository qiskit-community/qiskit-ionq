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

import re
import json
from collections import OrderedDict
from datetime import datetime
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
    def get_calibration_data(
        self, backend_name: str, limit: int | None = None
    ) -> Characterization | list[Characterization]:
        """Retrieve calibration data for a specified backend.

        Args:
            backend_name (str): The IonQ backend to fetch data for.
            limit (int, optional): Limit the number of results returned.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.
            IonQRetriableError: When a retriable error occurs during the request.

        Returns:
            Characterization: An instance of Characterization containing the calibration data
            or a list of Characterization instances if multiple results are returned.
        """
        params = {"limit": limit} if limit else None
        url = self.make_path("backends", backend_name, "characterizations")
        res = self.get_with_retry(url, headers=self.api_headers, params=params)
        exceptions.IonQAPIError.raise_for_status(res)
        chars = res.json().get("characterizations", [])
        return (
            Characterization(chars[0])
            if limit == 1
            else [Characterization(item) for item in chars]
        )

    @retry(exceptions=IonQRetriableError, max_delay=60, backoff=2, jitter=1)
    def get_results(
        self,
        results_url: str,
        sharpen: Optional[bool] = None,
        extra_query_params: Optional[dict] = None,
    ) -> dict:
        """Retrieve job results from the IonQ API.

        The returned JSON dict will only have data if job has completed.

        Args:
            results_url (str): The URL of the job results to retrieve.
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

        # Strip second API version (/v0.4/)
        req_path = re.sub(r"/v\d+\.\d+/", "", self.make_path(results_url), count=1)
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
    ) -> JobEstimate:
        """Call GET /jobs/estimate … returns a cost/time prediction."""
        params = {
            "type": job_type,
            "backend": backend.replace("ionq_qpu", "qpu"),
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
        return JobEstimate(res.json())

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


class Characterization:
    """
    Simple wrapper around the `/backends/<backend>/characterizations/<uuid>` payload.
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    # metadata
    @property
    def id(self) -> str:  # pylint: disable=invalid-name
        """UUID of this characterization."""
        return self._data["id"]

    @property
    def backend(self) -> str:
        """Backend name, e.g. `"qpu.aria-1"`."""
        return self._data["backend"]

    @property
    def date(self) -> datetime:
        """Timestamp of the measurement (UTC)."""
        return datetime.fromisoformat(self._data["date"].replace("Z", "+00:00"))

    # qubit info
    @property
    def qubits(self) -> int:
        """Number of qubits available."""
        return int(self._data["qubits"])

    @property
    def connectivity(self) -> list[tuple[int, int]]:
        """Valid two-qubit gate pairs as a list of tuples."""
        return [tuple(pair) for pair in self._data.get("connectivity", [])]

    # fidelity block
    @property
    def fidelity(self) -> dict:
        """Full fidelity dictionary (spam, 1q, 2q, ...)."""
        return self._data.get("fidelity", {})

    @property
    def median_spam_fidelity(self) -> float | None:  # convenience accessor
        """Median state-prep-and-measurement fidelity, if present."""
        return self.fidelity.get("spam", {}).get("median")

    # timing block
    @property
    def timing(self) -> dict:
        """Dictionary of timing parameters (readout, reset, 1q, 2q, t1, t2)."""
        return self._data.get("timing", {})

    def __repr__(self) -> str:
        parts = [
            f"backend={self.backend}",
            f"id={self.id}",
            f"date={self.date.isoformat()}",
            f"qubits={self.qubits}",
            f"connectivity={self.connectivity}",
            f"fidelity={self.fidelity}",
            f"timing={self.timing}",
        ]
        return f"Characterization({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Characterization) and other._data == self._data


class JobEstimate:
    """
    Wrapper for the payload returned by GET /jobs/estimate.
    """

    def __init__(self, data: dict):
        # we keep the original dict just in case (for .to_dict())
        self._raw = data

        # Flatten the interesting bits so they are attributes
        inputs = data.get("input_values", {})
        self.backend: str | None = inputs.get("backend")
        self.oneq_gates: int | None = inputs.get("1q_gates")
        self.twoq_gates: int | None = inputs.get("2q_gates")
        self.qubits: int | None = inputs.get("qubits")
        self.shots: int | None = inputs.get("shots")
        self.error_mitigation: bool = inputs.get("error_mitigation")
        self.session: bool = inputs.get("session")

        # Core numeric results
        self.cost: float | None = data.get("estimated_cost")
        self.cost_unit: str | None = data.get("cost_unit")
        self.exec_time: float | None = data.get("estimated_execution_time")  # seconds
        self.queue_time: float | None = data.get("current_predicted_queue_time")  # sec

        # When was this generated?
        ts = data.get("estimated_at")  # pylint: disable=invalid-name
        self.estimated_at: datetime | None = (
            datetime.fromisoformat(ts) if isinstance(ts, str) else None
        )

        # Optional structured pricing breakdown
        self.rate_information: dict = data.get("rate_information", {})

    # convenience helpers
    @property
    def total_runtime(self) -> float | None:
        """Predicted queue + execution time, in seconds (if both are present)."""
        if self.exec_time is None or self.queue_time is None:
            return None
        return self.exec_time + self.queue_time

    def to_dict(self) -> dict:
        """Return a shallow copy of the original JSON dict."""
        return dict(self._raw)

    def __repr__(self) -> str:
        parts = [
            f"backend={self.backend}",
            f"qubits={self.qubits}",
            f"shots={self.shots}",
            f"cost={self.cost} {self.cost_unit}",
            f"exec_time={self.exec_time}s",
            f"queue_time={self.queue_time}s",
        ]
        return f"JobEstimate({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, JobEstimate) and other._raw == self._raw


__all__ = ["IonQClient", "Characterization", "JobEstimate"]
