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

"""Basic API Client for IonQ's REST API.

Every outbound HTTP call to ``api.ionq.co`` is made through
``ionq-core`` (IonQ's official low-level Python client). This module
exposes the same legacy-shaped facade that the rest of ``qiskit-ionq``
already calls, so callers (``IonQJob``, ``IonQBackend``, ``Session``)
do not need to know about the transport switch.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from warnings import warn

import httpx
from ionq_core import ClientExtension, IonQClient as _IonQCoreClient
from ionq_core.api.characterizations import (
    get_characterizations_for_backend as _ionq_get_characterizations,
)
from ionq_core.api.default import (
    cancel_job as _ionq_cancel_job,
    create_job as _ionq_create_job,
    create_session as _ionq_create_session,
    delete_job as _ionq_delete_job,
    end_session as _ionq_end_session,
    estimate_job_cost as _ionq_estimate_job_cost,
    get_job as _ionq_get_job,
    get_job_probabilities as _ionq_get_job_probabilities,
)
from ionq_core.exceptions import APIError as _IonQCoreAPIError
from ionq_core.models import (
    CircuitJobCreationPayload,
    CreateSessionRequest,
    JSONMultiCircuitJob,
)
from ionq_core.types import UNSET

from . import exceptions
from .exceptions import IonQRetriableError
from .helpers import get_user_agent, qiskit_to_ionq, retry

if TYPE_CHECKING:  # pragma: no cover
    from .ionq_job import IonQJob


_DEFAULT_BASE_URL = "https://api.ionq.co/v0.4"


def _response_to_dict(response) -> dict:
    """Convert an ``ionq-core`` ``Response`` to the legacy dict-shaped return.

    Prefers the typed ``response.parsed`` model when ``ionq-core`` was able to
    coerce the body into the OpenAPI-modelled schema. Falls back to a raw
    ``json.loads`` of ``response.content`` when the body has fields the typed
    model rejects (legacy partial-payload tests, forward-compatible API
    additions, etc.) so the caller gets the same dict that ``requests``-based
    code paths used to return.
    """
    parsed = getattr(response, "parsed", None)
    if parsed is not None and hasattr(parsed, "to_dict"):
        return parsed.to_dict()
    raw = getattr(response, "content", None)
    if raw:
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return {}


def _call_through_core(endpoint_call, /, **kwargs):
    """Run an ``ionq-core`` ``sync_detailed`` and tolerate strict-parse misses.

    ``ionq-core`` 0.1.x's OpenAPI-generated ``from_dict`` models pop required
    keys eagerly, so any test fixture that builds a partial mock job/results
    payload (the qiskit-ionq suite has many of these) raises ``KeyError`` from
    inside the parser. Production responses from ``api.ionq.co`` always
    include every required field, so this fallback only matters for tests.
    When parsing fails we re-issue the request via the underlying httpx
    client and return a synthetic ``Response``-shaped object whose
    ``content`` is the raw bytes - ``_response_to_dict`` then falls back to
    ``json.loads``.

    TODO(ionq-core): transitional. Drop this helper (and ``_BareResponse``
    below) once test fixtures are migrated to typed ``ionq-core`` models, or
    once ``ionq-core`` ships a lenient parser mode. Tracked alongside the
    rest of the ``qiskit_ionq.exceptions`` migration.
    """
    try:
        return endpoint_call(**kwargs)
    except (KeyError, ValueError, TypeError):
        client = kwargs.get("client")
        if client is None:
            raise
        import importlib

        get_kwargs = importlib.import_module(endpoint_call.__module__)._get_kwargs
        request_kwargs = {k: v for k, v in kwargs.items() if k != "client"}
        http_kwargs = get_kwargs(**request_kwargs)
        response = client.get_httpx_client().request(**http_kwargs)
        return _BareResponse(response.status_code, response.content)


class _BareResponse:
    """Minimal stand-in for ``ionq_core.types.Response`` when typed parsing
    failed. See ``_call_through_core``. Transitional."""

    parsed = None

    def __init__(self, status_code: int, content: bytes):
        self.status_code = status_code
        self.content = content


def _raise_status(response, raise_retriable: bool) -> None:
    """Translate a non-2xx ``ionq-core`` response into the legacy exceptions."""
    status = response.status_code
    if 200 <= status < 300:
        return
    body_text = (
        response.content.decode()
        if isinstance(response.content, (bytes, bytearray))
        else str(response.content)
    )
    err = exceptions.IonQAPIError.from_ionq_core(
        _IonQCoreAPIError(status_code=status, body=body_text, message=body_text)
    )
    if raise_retriable and exceptions._is_retriable("GET", status):
        raise IonQRetriableError(err)
    raise err


class IonQClient:
    """IonQ API Client backed by ``ionq-core``.

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
        # ``ionq-core`` is the sole HTTP transport. ``api_key`` is required at
        # construction time; the legacy ``IonQClient()`` with no token also
        # constructed successfully and only failed on the first wire call, so
        # use a placeholder here to preserve that pattern (real calls without
        # a real key will surface a 401 from the server, same as before).
        self._core = _IonQCoreClient(
            api_key=token or os.environ.get("IONQ_API_KEY") or "MISSING_API_KEY",
            base_url=self._url or _DEFAULT_BASE_URL,
            extension=ClientExtension(
                user_agent_token=self._user_agent,
                default_headers=dict(self._custom_headers),
            ),
        )

    @property
    def api_headers(self) -> dict:
        """API Headers needed to make calls to the REST API.

        Returns:
            dict[str, str]: A dict of request headers.
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
            httpx.Response: The transport-level response. Callers should
                normally use the typed ``IonQClient`` methods instead of
                touching this helper directly.
        """
        try:
            res = self._core.get_httpx_client().get(
                req_path, params=params, headers=headers, timeout=timeout
            )
        except httpx.HTTPError as exc:
            raise IonQRetriableError(exc) from exc
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
            dict: The API response as a dict.
        """
        as_json = qiskit_to_ionq(
            job.circuit,
            job.backend(),
            job._passed_args,
            job.extra_query_params,
            job.extra_metadata,
        )
        body_dict = json.loads(as_json)
        body_cls = (
            JSONMultiCircuitJob
            if body_dict.get("type") == "ionq.multi-circuit.v1"
            else CircuitJobCreationPayload
        )
        body = body_cls.from_dict(body_dict)
        try:
            response = _call_through_core(
                _ionq_create_job.sync_detailed, client=self._core, body=body
            )
        except _IonQCoreAPIError as exc:
            raise exceptions.IonQAPIError.from_ionq_core(exc) from exc
        _raise_status(response, raise_retriable=True)
        return _response_to_dict(response)

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
            dict: The API response as a dict.
        """
        try:
            response = _call_through_core(
                _ionq_get_job.sync_detailed, uuid=job_id, client=self._core
            )
        except _IonQCoreAPIError as exc:
            raise exceptions.IonQAPIError.from_ionq_core(exc) from exc
        _raise_status(response, raise_retriable=True)
        return _response_to_dict(response)

    @retry(exceptions=IonQRetriableError, tries=5)
    def cancel_job(self, job_id: str) -> dict:
        """Attempt to cancel a job which has not yet run.

        .. NOTE:: If the job has already reached status "completed", this cancel action is a no-op.

        Args:
            job_id (str): The ID of the job to cancel.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.

        Returns:
            dict: The API response as a dict.
        """
        try:
            response = _call_through_core(
                _ionq_cancel_job.sync_detailed, uuid=job_id, client=self._core
            )
        except _IonQCoreAPIError as exc:
            raise exceptions.IonQAPIError.from_ionq_core(exc) from exc
        _raise_status(response, raise_retriable=True)
        return _response_to_dict(response)

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
            dict: The API response as a dict.
        """
        try:
            response = _call_through_core(
                _ionq_delete_job.sync_detailed, uuid=job_id, client=self._core
            )
        except _IonQCoreAPIError as exc:
            raise exceptions.IonQAPIError.from_ionq_core(exc) from exc
        _raise_status(response, raise_retriable=True)
        return _response_to_dict(response)

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
        try:
            response = _ionq_get_characterizations.sync_detailed(
                backend=backend_name,  # ty: ignore[invalid-argument-type]
                client=self._core,
                limit=limit if limit is not None else UNSET,
            )
        except _IonQCoreAPIError as exc:
            raise exceptions.IonQAPIError.from_ionq_core(exc) from exc
        _raise_status(response, raise_retriable=True)
        chars = _response_to_dict(response).get("characterizations", [])
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
            results_url (str): The URL of the job results to retrieve. The
                ``ionq-core`` ``get_job_probabilities`` endpoint takes a job
                UUID, which is extracted from the trailing
                ``jobs/<uuid>/results/probabilities`` segment.
            sharpen (bool): Supported if the job is debiased,
            allows you to filter out physical qubit bias from the results.
            extra_query_params (dict): Specify any parameters to include in the request

        Raises:
            IonQAPIError: When the API returns a non-200 status code.
            IonQRetriableError: When a retriable error occurs during the request.
            IonQClientError: When ``results_url`` cannot be parsed for a job UUID.

        Returns:
            dict: A :mod:`requests <requests>` response :meth:`json <requests.Response.json>` dict.
        """
        # Pull the job UUID out of the legacy ``/v0.4/jobs/<uuid>/results/...``
        # path so the typed ``ionq-core`` endpoint can be addressed directly.
        parts = [p for p in results_url.strip("/").split("/") if p]
        try:
            job_uuid = parts[parts.index("jobs") + 1]
        except (ValueError, IndexError) as exc:  # pragma: no cover - guard
            raise exceptions.IonQClientError(
                f"Could not parse job UUID from results URL {results_url!r}"
            ) from exc

        # Build the query-string dict the way the legacy implementation did.
        params: dict = {}
        if sharpen is not None:
            params["sharpen"] = sharpen
        if extra_query_params:
            warn(
                f"The parameter(s): {extra_query_params} is not checked by default "
                "but will be submitted in the request."
            )
            params.update(extra_query_params)

        # If the caller stuck to ``sharpen``, prefer the typed ionq-core
        # endpoint. Otherwise fall back to the underlying httpx client (still
        # owned by ionq-core) so unmodelled escape-hatch query params survive.
        if extra_query_params:
            url = (
                self._core.get_httpx_client()
                .base_url.copy_with()
                .join(f"jobs/{job_uuid}/results/probabilities")
            )
            response = self._core.get_httpx_client().get(str(url), params=params)
        else:
            try:
                response = _ionq_get_job_probabilities.sync_detailed(
                    uuid=job_uuid,
                    client=self._core,
                    sharpen=sharpen if sharpen is not None else UNSET,
                )
            except _IonQCoreAPIError as exc:
                raise exceptions.IonQAPIError.from_ionq_core(exc) from exc
        _raise_status(response, raise_retriable=True)
        # Re-parse the raw bytes through ``OrderedDict`` to preserve key order
        # for callers that depend on insertion-order iteration of probabilities.
        return json.loads(response.content, object_pairs_hook=OrderedDict)

    def estimate_job(
        self,
        *,
        backend: str,
        oneq_gates: int,
        twoq_gates: int,
        qubits: int,
        shots: int,
        error_mitigation: bool = False,
        session: bool = False,  # pylint: disable=unused-argument
        job_type: str = "ionq.circuit.v1",
    ) -> JobEstimate:
        """Call ``GET /jobs/estimate`` and return a cost/time prediction.

        ``session`` is accepted for backwards compatibility with the legacy
        signature; the v0.4 ``estimate_job_cost`` endpoint no longer takes it.
        """
        try:
            response = _ionq_estimate_job_cost.sync_detailed(
                client=self._core,
                backend=backend.replace("ionq_qpu", "qpu"),
                type_=job_type,
                qubits=qubits,
                shots=shots,
                field_1q_gates=oneq_gates,
                field_2q_gates=twoq_gates,
                error_mitigation=error_mitigation,
            )
        except _IonQCoreAPIError as exc:
            raise exceptions.IonQAPIError.from_ionq_core(exc) from exc
        _raise_status(response, raise_retriable=False)
        return JobEstimate(_response_to_dict(response))

    @retry(exceptions=IonQRetriableError, tries=5)
    def post(self, *path_parts: str, json_body: dict | None = None) -> dict:
        """POST helper used by ``Session`` lifecycle endpoints.

        Routes the four legacy session paths
        (``/sessions``, ``/sessions/<id>/end``) through the corresponding
        typed ``ionq-core`` endpoints. Any other path raises
        ``IonQClientError`` - we do not provide a generic POST escape
        hatch for arbitrary URLs because every supported call has a
        typed ``ionq-core`` equivalent.

        Args:
            *path_parts (str): Path parts to append to the base URL.
            json_body (dict, optional): JSON body to send in the POST request.

        Raises:
            IonQAPIError: When the API returns a non-200 status code.
            IonQRetriableError: When a retriable error occurs during the request.
            IonQClientError: When ``path_parts`` does not name a known IonQ endpoint.

        Returns:
            dict: The API response as a dict.
        """
        try:
            if path_parts == ("sessions",):
                response = _ionq_create_session.sync_detailed(
                    client=self._core,
                    body=CreateSessionRequest.from_dict(json_body or {}),
                )
            elif (
                len(path_parts) == 3
                and path_parts[0] == "sessions"
                and path_parts[2] == "end"
            ):
                response = _ionq_end_session.sync_detailed(
                    session_id=path_parts[1], client=self._core
                )
            else:
                raise exceptions.IonQClientError(
                    f"POST to {'/'.join(path_parts)!r} is not a known IonQ endpoint."
                )
        except _IonQCoreAPIError as exc:
            raise exceptions.IonQAPIError.from_ionq_core(exc) from exc
        _raise_status(response, raise_retriable=True)
        return _response_to_dict(response)

    @retry(exceptions=IonQRetriableError, tries=3)
    def put(self, *path_parts: str, json_body: dict | None = None) -> dict:  # pylint: disable=unused-argument
        """Legacy PUT helper. The v0.4 surface no longer requires arbitrary PUTs;
        every endpoint that previously needed one (``/jobs/<id>/status/cancel``)
        now has a typed ``ionq-core`` equivalent reached via :meth:`cancel_job`.

        Args:
            *path_parts (str): Path parts to append to the base URL.
            json_body (dict, optional): Ignored on the new transport; kept for
                signature parity with the legacy ``IonQClient.put`` helper.

        Raises:
            IonQClientError: When ``path_parts`` does not name a known IonQ endpoint.
        """
        if (
            len(path_parts) == 4
            and path_parts[0] == "jobs"
            and path_parts[2:] == ("status", "cancel")
        ):
            return self.cancel_job(path_parts[1])
        raise exceptions.IonQClientError(
            f"PUT to {'/'.join(path_parts)!r} is not a known IonQ endpoint."
        )


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
    def status(self) -> str:
        """Status of the characterization, e.g. `"available"`."""
        return self._data["status"]

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
