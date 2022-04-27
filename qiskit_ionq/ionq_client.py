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

import requests

from . import exceptions
from .helpers import qiskit_to_ionq


class IonQClient:
    """IonQ API Client

    Attributes:
        _url(str): A URL base to use for API calls, e.g. ``"https://api.ionq.co/v0.1"``
        _token(str): An API Access Token to use with the IonQ API.
    """

    def __init__(self, token=None, url=None):
        self._token = token
        # strip trailing slashes from our base URL.
        if url and url.endswith("/"):
            url = url[:-1]
        self._url = url

    @property
    def api_headers(self):
        """API Headers needed to make calls to the REST API.

        Returns:
            dict[str, str]: A dict of :class:`requests.Request` headers.
        """
        return {
            "Authorization": f"apiKey {self._token}",
            "Content-Type": "application/json",
        }

    def make_path(self, *parts):
        """Make a "/"-delimited path, then append it to :attr:`_url`.

        Returns:
            str: A URL to use for an API call.
        """
        return "/".join([self._url] + list(parts))

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
            job.circuit, job.backend().name(), job.backend().lang(), job._passed_args
        )
        req_path = self.make_path("jobs")
        res = requests.post(req_path, data=as_json, headers=self.api_headers)
        if res.status_code != 200:
            raise exceptions.IonQAPIError.from_response(res)
        return res.json()

    def retrieve_job(self, job_id: str):
        """Retrieve job information from the IonQ API.

        The returned JSON dict will only have data if job has completed.

        Args:
            job_id (str): The ID of a job to retrieve.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """

        req_path = self.make_path("jobs", job_id)
        res = requests.get(req_path, headers=self.api_headers)
        if res.status_code != 200:
            raise exceptions.IonQAPIError.from_response(res)
        return res.json()

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
        res = requests.put(req_path, headers=self.api_headers)
        if res.status_code != 200:
            raise exceptions.IonQAPIError.from_response(res)
        return res.json()

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
        res = requests.delete(req_path, headers=self.api_headers)
        if res.status_code != 200:
            raise exceptions.IonQAPIError.from_response(res)
        return res.json()

    def get_calibration_data(self, backend_name: str) -> dict:
        """Retrieve calibration data for a specified backend.

        Args:
            backend_name (str): The IonQ backend to fetch data for.

        Calibration::

            {
                "pages": <int>,
                "calibrations": [
                    {
                        "id": <str>,
                        "date": <int>,
                        "target": <str>,
                        "qubits": <int>,
                        "connectivity": [<int>, ...],
                        "fidelity": {
                            "spam": {
                                "mean": <int>,
                                "stderr": <int>
                            }
                        },
                        "timing": {
                            "readout": <int>,
                            "reset": <int>
                        }
                    }
                ]
            }

        Returns:
            dict: A dictionary of an IonQ backend's calibration data.
        """
        req_path = self.make_path("calibrations")
        res = requests.get(req_path, headers=self.api_headers)
        if res.status_code != 200:
            raise exceptions.IonQAPIError.from_response(res)

        # Get calibrations and filter down to the target
        response = res.json()
        calibrations = response.get("calibrations") or []
        calibrations = [c for c in calibrations if c.get("target") == backend_name]

        # If nothing was found, just return None.
        if len(calibrations) == 0:
            return None

        # Calibrations are in most recent order in the response, so get the first.
        return calibrations[0]


__all__ = ["IonQClient"]
