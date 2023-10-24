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

from typing import Optional
from warnings import warn
import requests

from retry import retry

from . import exceptions
from .helpers import qiskit_to_ionq, get_user_agent
from .exceptions import IonQRetriableError


class IonQClient:
    """IonQ API Client

    Attributes:
        _url(str): A URL base to use for API calls, e.g. ``"https://api.ionq.co/v0.3"``
        _token(str): An API Access Token to use with the IonQ API.
        _custom_headers(dict): Extra headers to add to the request.
    """

    def __init__(self, token=None, url=None, custom_headers=None):
        self._token = token
        self._custom_headers = custom_headers or {}
        # strip trailing slashes from our base URL.
        if url and url.endswith("/"):
            url = url[:-1]
        self._url = url
        self._user_agent = get_user_agent()

    @property
    def api_headers(self):
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

    def make_path(self, *parts):
        """Make a "/"-delimited path, then append it to :attr:`_url`.

        Returns:
            str: A URL to use for an API call.
        """
        return "/".join([self._url] + list(parts))

    def _get_with_retry(self, req_path, params=None, headers=None, timeout=30):
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
                req_path, params=params, headers=headers, timeout=timeout
            )
        except requests.exceptions.RequestException as req_exc:
            raise IonQRetriableError(req_exc) from req_exc

        return res

    @retry(exceptions=IonQRetriableError, tries=5)
    def submit_job(self, job) -> dict:
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
    def retrieve_job(self, job_id: str):
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
        res = self._get_with_retry(req_path, headers=self.api_headers)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    @retry(exceptions=IonQRetriableError, tries=5)
    def cancel_job(self, job_id: str):
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

    @retry(exceptions=IonQRetriableError, tries=3)
    def delete_job(self, job_id: str):
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
    def get_calibration_data(self, backend_name: str) -> dict:
        """Retrieve calibration data for a specified backend.

        Args:
            backend_name (str): The IonQ backend to fetch data for.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.
            IonQRetriableError: When a retriable error occurs during the request.

        Returns:
            dict: A dictionary of an IonQ backend's calibration data.
        """
        req_path = self.make_path(
            "/".join(["characterizations/backends", backend_name[5:], "current"])
        )
        res = self._get_with_retry(req_path, headers=self.api_headers)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()

    @retry(exceptions=IonQRetriableError, max_delay=60, backoff=2, jitter=1)
    def get_results(
        self,
        job_id: str,
        sharpen: Optional[bool] = None,
        extra_query_params: Optional[dict] = None,
    ):
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

        req_path = self.make_path("jobs", job_id, "results")
        res = self._get_with_retry(req_path, headers=self.api_headers, params=params)
        exceptions.IonQAPIError.raise_for_status(res)
        return res.json()


__all__ = ["IonQClient"]
