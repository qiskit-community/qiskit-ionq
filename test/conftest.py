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

"""global pytest fixtures"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager

import pytest

from qiskit_ionq import ionq_backend, ionq_job, ionq_provider
from qiskit_ionq.helpers import compress_to_metadata_string


# ---------------------------------------------------------------------------
# pytest-httpx → requests_mock shim
# ---------------------------------------------------------------------------
# qiskit-ionq used to ship its own ``requests``-based HTTP client and the test
# suite mocks the wire with ``requests-mock``. Now that every endpoint runs
# through ``ionq-core`` (which is ``httpx``-backed), ``requests-mock`` cannot
# intercept the wire traffic. ``pytest-httpx`` is the canonical mock library
# for ``httpx``; the thin shim below preserves the ``requests_mock`` API
# (``.get(url, json=...)``, ``.post(...)``, ``.request_history[*].json()``)
# so the existing test files do not need to change.


class _HistoryItem:
    """Adapter exposing the ``requests_mock`` history-item API on top of an
    ``httpx.Request``."""

    def __init__(self, request):
        self._request = request
        self.method = request.method
        self.url = str(request.url)

    def json(self):
        """Parse the captured request body as JSON (matches ``requests_mock``)."""
        return json.loads(self._request.content)


class _RequestsMockShim:
    """Translate the small ``requests_mock`` surface used in the test suite
    onto pytest-httpx's ``httpx_mock`` fixture."""

    def __init__(self, httpx_mock):
        self._mock = httpx_mock

    def _register(self, method, url, **kwargs):
        """Register a mocked response for ``method`` on ``url``."""
        # ``requests_mock`` matches the URL path without considering the query
        # string by default; ``httpx_mock`` is exact-string by default. Wrap
        # plain URLs in a regex that also accepts ``?...`` suffixes so existing
        # tests (e.g. ``?sharpen=true``) keep matching their base mocks.
        if isinstance(url, str) and "?" not in url:
            url = re.compile(re.escape(url) + r"(\?.*)?$")
        kwargs.setdefault("status_code", 200)
        # ``is_reusable=True`` mirrors requests-mock's behaviour where the same
        # URL can be hit multiple times in one test (e.g. polling job.status).
        self._mock.add_response(method=method, url=url, is_reusable=True, **kwargs)

    def get(self, url, **kwargs):
        """Register a GET response (legacy ``requests_mock.get`` shape)."""
        self._register("GET", url, **kwargs)

    def post(self, url, **kwargs):
        """Register a POST response (legacy ``requests_mock.post`` shape)."""
        self._register("POST", url, **kwargs)

    def put(self, url, **kwargs):
        """Register a PUT response (legacy ``requests_mock.put`` shape)."""
        self._register("PUT", url, **kwargs)

    def delete(self, url, **kwargs):
        """Register a DELETE response (legacy ``requests_mock.delete`` shape)."""
        self._register("DELETE", url, **kwargs)

    def register_uri(self, method, url, **kwargs):
        """Register a response, supporting the legacy ``ANY`` sentinel."""
        if url is _ANY:
            url = re.compile(r".*")
        if method is _ANY:
            for m in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                self._register(m, url, **kwargs)
            return
        self._register(method, url, **kwargs)

    @property
    def request_history(self):
        """List of recorded requests (legacy ``requests_mock.request_history``).

        Filter out the unauthenticated ``GET /backends/<name>`` traffic that
        every backend now issues at construction time via
        ``helpers.get_n_qubits``. Tests written against the legacy
        ``requests``-based client never saw those requests, so preserving the
        old history shape keeps existing assertions on history length valid.
        """
        return [
            _HistoryItem(r)
            for r in self._mock.get_requests()
            if "/backends/" not in str(r.url)
        ]


# ANY sentinels that match ``requests_mock.adapter.ANY``.
_ANY = object()


@contextmanager
def _default_requests_mock(httpx_mock):
    """Context-manager shim used by ``formatted_result``.

    The legacy form was ``with _default_requests_mock() as requests_mock:`` and
    used a fresh per-context mocker. With pytest-httpx the lifetime is owned
    by the test fixture; this just yields a shim that writes into the
    surrounding ``httpx_mock``.

    Args:
        httpx_mock (HTTPXMock): The pytest-httpx fixture instance.
    """
    yield _RequestsMockShim(httpx_mock)


@pytest.fixture(autouse=True)
def _httpx_mock_autouse(httpx_mock):
    """Activate ``pytest-httpx`` for every test.

    qiskit-ionq now talks to the IonQ API exclusively through ``ionq-core``
    (and therefore ``httpx``). ``httpx_mock`` only intercepts requests when
    a test references the fixture; declaring it autouse here prevents any
    test from accidentally hitting ``api.ionq.co`` (which would hang the
    job in CI). This mirrors the ``pytest_sessionstart`` ``register_uri(ANY,
    ANY, status_code=599)`` behaviour the legacy ``requests-mock`` conftest
    used.

    The ``get_n_qubits`` helper is ``lru_cache``-decorated so production
    callers don't re-issue redundant ``GET /backends/<name>`` calls; that
    cache survives across tests and would otherwise pollute the suite if
    any test happened to populate it before httpx_mock was active. Clear
    it at the top of every test so each test sees a clean slate.

    Args:
        httpx_mock (HTTPXMock): The pytest-httpx fixture instance.
    """
    from qiskit_ionq.helpers import get_n_qubits

    get_n_qubits.cache_clear()
    return httpx_mock


@pytest.fixture
def requests_mock(httpx_mock):
    """A ``requests_mock``-compatible fixture backed by ``pytest-httpx``.

    Args:
        httpx_mock (HTTPXMock): The pytest-httpx fixture instance.

    Returns:
        _RequestsMockShim: An object exposing the subset of the
            ``requests_mock`` API used by the test suite.
    """
    return _RequestsMockShim(httpx_mock)


def pytest_collection_modifyitems(items):
    """Match the lenient assertion behaviour of the legacy ``requests-mock`` suite.

    The legacy global ``register_uri`` returned a 599 instead of failing fast,
    so registered responses didn't all need to be hit and unmocked requests
    didn't fail the test. Mirror that on ``pytest-httpx``.

    Args:
        items (list): The collected pytest items.
    """
    for item in items:
        item.add_marker(
            pytest.mark.httpx_mock(
                assert_all_responses_were_requested=False,
                assert_all_requests_were_expected=False,
            )
        )


def _def_results_template(job_id):
    """A template for the results field in a job response."""
    return {
        "probabilities": {
            # v0.4 returns a relative path - the client prefixes it with the base URL
            # https://docs.ionq.com/api-reference/v0.4/jobs/get-job
            "url": f"/v0.4/jobs/{job_id}/results/probabilities"
        }
    }


class MockBackend(ionq_backend.IonQBackend):
    """A mock backend for testing super-class behavior in isolation."""

    def __init__(self, provider, *, name: str = "ionq_mock_backend"):  # pylint: disable=redefined-outer-name
        """
        Build a minimal mock backend that satisfies BackendV2.
        """
        super().__init__(
            provider=provider,
            name=name,
            description="IonQ Mock Backend",
            gateset="qis",
            num_qubits=11,
            simulator=True,
            max_shots=10_000,
        )


def dummy_job_response(
    job_id, target="mock_backend", status="completed", job_settings=None, children=None
):
    """A dummy response payload for `job_id`.

    Args:
        job_id (str): An arbitrary job id.
        target (str): Backend target string.
        status (str): A provided status string.
        job_settings (dict): Settings provided to the API.
        children (list): A list of child job IDs.

    Returns:
        dict: A json response dict.
    """
    qiskit_header = compress_to_metadata_string(
        {
            "qubit_labels": [["q", 0], ["q", 1]],
            "n_qubits": 2,
            "qreg_sizes": [["q", 2]],
            "clbit_labels": [["c", 0], ["c", 1]],
            "memory_slots": 2,
            "creg_sizes": [["c", 2]],
            "name": job_id,
            "global_phase": 0,
        }
    )
    response = {
        "status": status,
        "predicted_execution_time": 4,
        "metadata": {
            "qobj_id": "test_qobj_id",
            "shots": "1234",
            "sampler_seed": "42",
            "output_length": "2",
            "qiskit_header": qiskit_header,
        },
        "execution_time": 8,
        "qubits": 2,
        "type": "circuit",
        "request": 1600000000,
        "start": 1600000001,
        "response": 1600000002,
        "backend": target,
        "results": _def_results_template(job_id),
        "id": job_id,
        "settings": (job_settings or {}),
        "name": "test_name",
    }

    if children is not None:
        response["children"] = children

    return response


def dummy_mapped_job_response(
    job_id, target="mock_backend", status="completed", job_settings=None, children=None
):
    """A dummy mapped response payload for `job_id`.

    Args:
        job_id (str): An arbitrary job id.
        target (str): Backend target string.
        status (str): A provided status string.
        job_settings (dict): Settings provided to the API.
        children (list): A list of child job IDs.

    Returns:
        dict: A json response dict.
    """
    qiskit_header = compress_to_metadata_string(
        {
            "qubit_labels": [["q", 0], ["q", 1]],
            "n_qubits": 2,
            "qreg_sizes": [["q", 2]],
            "clbit_labels": [["c", 0], ["c", 1]],
            "memory_slots": 2,
            "creg_sizes": [["c", 2]],
            "name": job_id,
            "global_phase": 0,
            "meas_mapped": [1, 0],
        }
    )
    response = {
        "status": status,
        "predicted_execution_time": 4,
        "metadata": {
            "qobj_id": "test_qobj_id",
            "shots": "1234",
            "sampler_seed": "42",
            "output_length": "2",
            "qiskit_header": qiskit_header,
        },
        "execution_time": 8,
        "qubits": 2,
        "type": "circuit",
        "request": 1600000000,
        "start": 1600000001,
        "response": 1600000002,
        "backend": target,
        "results": _def_results_template(job_id),
        "id": job_id,
        "settings": (job_settings or {}),
        "name": "test_name",
    }

    if children is not None:
        response["children"] = children

    return response


def dummy_failed_job(job_id):  # pylint: disable=differing-param-doc,differing-type-doc
    """A dummy response payload for a failed job.

    Args:
        job_id (str): An arbitrary job id.
        status (str): A provided status string.

    Returns:
        dict: A json response dict.

    """
    qiskit_header = compress_to_metadata_string(
        {
            "qubit_labels": [["q", 0], ["q", 1]],
            "n_qubits": 2,
            "qreg_sizes": [["q", 2]],
            "clbit_labels": [["c", 0], ["c", 1]],
            "memory_slots": 2,
            "creg_sizes": [["c", 2]],
            "name": job_id,
            "global_phase": 0,
        }
    )
    return {
        "failure": {"error": "example error", "code": "ExampleError"},
        "status": "failed",
        "metadata": {"shots": "1", "qiskit_header": qiskit_header},
        "type": "circuit",
        "request": 1600000000,
        "response": 1600000002,
        "backend": "qpu",
        "results": _def_results_template(job_id),
        "id": job_id,
    }


@pytest.fixture()
def provider():
    """Fixture for injecting a test provider.

    Returns:
        IonQProvider: A provider suitable for testing.
    """
    return ionq_provider.IonQProvider("token")


@pytest.fixture()
def mock_backend(provider):  # pylint: disable=redefined-outer-name
    """A fixture instance of the :class:`MockBackend`.

    Args:
        provider (IonQProvider): An IonQProvider fixture.

    Returns:
        MockBackenbd: An instance of :class:`MockBackend`
    """
    return MockBackend(provider)


# pylint: disable=redefined-outer-name
@pytest.fixture()
def qpu_backend(provider):
    """Get the QPU backend from a provider.

    Args:
        provider (IonQProvider): Injected provider from :meth:`provider`.

    Returns:
        IonQQPUBackend: An instance of an IonQQPUBackend.
    """
    return provider.get_backend("ionq_qpu")


# pylint: disable=redefined-outer-name
@pytest.fixture()
def simulator_backend(provider):
    """Get the QPU backend from a provider.

    Args:
        provider (IonQProvider): Injected provider from :meth:`provider`.

    Returns:
        IonQQPUBackend: An instance of an IonQQPUBackend.
    """
    return provider.get_backend("ionq_simulator")


# pylint: disable=redefined-outer-name
@pytest.fixture()
def formatted_result(provider, httpx_mock):
    """Fixture for auto-injecting a formatted IonQJob result object into a
    a sub-class of ``unittest.TestCase``.

    Args:
        provider (IonQProvider): Injected provider from :meth:`provider`.
        httpx_mock (HTTPXMock): pytest-httpx fixture used to stub the API calls.

    Returns:
        Result: A qiskit result from making a fake API call with StubbedClient.
    """
    # Dummy job ID for formatted results fixture.
    job_id = "test_id"
    settings = {"lorem": {"ipsum": "dolor"}}

    # Create a backend and client to use for accessing the job.
    backend = provider.get_backend("ionq_qpu.aria-1")
    backend.set_options(job_settings=settings)
    client = backend._create_client()

    # Create the request path for accessing the dummy job:
    path = client.make_path("jobs", job_id)
    results_path = client.make_path("jobs", job_id, "results", "probabilities")

    # mock a job response
    with _default_requests_mock(httpx_mock) as requests_mock:
        # Mock the response with our dummy job response.
        requests_mock.get(
            path, json=dummy_job_response(job_id, "qpu.aria-1", "completed", settings)
        )

        requests_mock.get(results_path, json={"0": 0.5, "2": 0.499999})

        # Create the job (this calls self.status(), which will fetch the job).
        job = ionq_job.IonQJob(backend, job_id, client)

        # Yield so that the mock context manager properly unwinds.
        yield job.result()
