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
import pytest
import requests_mock as _requests_mock
from qiskit.providers import models as q_models
from requests_mock import adapter as rm_adapter

from qiskit_ionq import ionq_backend, ionq_job, ionq_provider
from qiskit_ionq.helpers import compress_dict_to_metadata_string


class MockBackend(ionq_backend.IonQBackend):
    """A mock backend for testing super-class behavior in isolation."""

    def gateset(self):
        return "qis"

    def __init__(self, provider, name="ionq_mock_backend"):  # pylint: disable=redefined-outer-name
        config = q_models.BackendConfiguration.from_dict(
            {
                "backend_name": name,
                "backend_version": "0.0.1",
                "simulator": True,
                "local": True,
                "coupling_map": None,
                "description": "IonQ Mock Backend",
                "n_qubits": 29,
                "conditional": False,
                "open_pulse": False,
                "memory": False,
                "max_shots": 0,
                "basis_gates": [],
                "gates": [
                    {
                        "name": "TODO",
                        "parameters": [],
                        "qasm_def": "TODO",
                    }
                ],
            }
        )
        super().__init__(config, provider=provider)

    def with_name(self, name, **kwargs):
        """Helper method that returns this backend with a more specific target system."""
        return MockBackend(self._provider, name, **kwargs)


def dummy_job_response(job_id, target="mock_backend", status="completed", job_settings=None):
    """A dummy response payload for `job_id`.

    Args:
        job_id (str): An arbitrary job id.
        target (str): Backend target string.
        status (str): A provided status string.
        job_settings (dict): Settings provided to the API.

    Returns:
        dict: A json response dict.
    """
    qiskit_header = compress_dict_to_metadata_string(
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
        "status": status,
        "predicted_execution_time": 4,
        "metadata": {
            "qobj_id": "test_qobj_id",
            "shots": "1234",
            "sampler_seed": "42",
            "output_length": "2",
            "qiskit_header": qiskit_header,
        },
        "registers": {"meas_mapped": [0, 1]},
        "execution_time": 8,
        "qubits": 2,
        "type": "circuit",
        "request": 1600000000,
        "start": 1600000001,
        "response": 1600000002,
        "target": target,
        "id": job_id,
        "settings": (job_settings or {}),
        "name": "test_name",
    }


def dummy_failed_job(job_id):  # pylint: disable=differing-param-doc,differing-type-doc
    """A dummy response payload for a failed job.

    Args:
        job_id (str): An arbitrary job id.
        status (str): A provided status string.

    Returns:
        dict: A json response dict.

    """
    qiskit_header = compress_dict_to_metadata_string(
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
        "target": "qpu",
        "id": job_id,
    }


def _default_requests_mock(**kwargs):
    """Create a default `requests_mock.Mocker` for use in tests.

    Args:
        kwargs (dict): Any additional kwargs to create the mocker with.

    Returns:
        :class:`request_mock.Mocker`: A requests mocker.
    """
    mocker_kwargs = {"real_http": False, **kwargs}
    mocker = _requests_mock.Mocker(**mocker_kwargs)
    return mocker


def pytest_sessionstart(session):
    """pytest hook for global test session start

    Args:
        session (:class:`pytest.Session`): A pytest session object.
    """
    session.global_requests_mock = _default_requests_mock()
    session.global_requests_mock.start()
    session.global_requests_mock.register_uri(
        rm_adapter.ANY,
        rm_adapter.ANY,
        status_code=599,
        text="UNHANDLED REQUEST. PLEASE MOCK WITH requests_mock.",
    )


def pytest_sessionfinish(session):
    """pytest hook for global test session end

    Args:
        session (:class:`pytest.Session`): A pytest session object.
    """
    session.global_requests_mock.stop()
    del session.global_requests_mock


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
def formatted_result(provider):
    """Fixture for auto-injecting a formatted IonQJob result object into a
    a sub-class of ``unittest.TestCase``.

    Args:
        provider (IonQProvider): Injected provider from :meth:`provider`.

    Returns:
        Result: A qiskit result from making a fake API call with StubbedClient.
    """
    # Dummy job ID for formatted results fixture.
    job_id = "test_id"
    settings = {"lorem": {"ipsum": "dolor"}}

    # Create a backend and client to use for accessing the job.
    backend = provider.get_backend("ionq_qpu.aria.1")
    backend.set_options(job_settings=settings)
    client = backend.create_client()

    # Create the request path for accessing the dummy job:
    path = client.make_path("jobs", job_id)
    results_path = client.make_path("jobs", job_id, "results")

    # mock a job response
    with _default_requests_mock() as requests_mock:
        # Mock the response with our dummy job response.
        requests_mock.get(path,
                          json=dummy_job_response(job_id, "qpu.aria.1", "completed", settings))

        requests_mock.get(results_path,
                          json={"0": 0.5, "2": 0.499999})

        # Create the job (this calls self.status(), which will fetch the job).
        job = ionq_job.IonQJob(backend, job_id, client)

        # Yield so that the mock context manager properly unwinds.
        yield job.result()
