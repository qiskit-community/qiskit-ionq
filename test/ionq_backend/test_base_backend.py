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

"""Tests for the IonQ's Backend base/super-class."""
# pylint: disable=redefined-outer-name

from unittest import mock

import pytest
from qiskit import QuantumCircuit
from qiskit.providers.models.backendstatus import BackendStatus

from qiskit_ionq import exceptions, ionq_client, ionq_job

from .. import conftest


def test_status_dummy_response(mock_backend):
    """Test that the IonQBackend returns an aribtrary backend status.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
    """
    status = mock_backend.status()
    assert isinstance(status, BackendStatus)
    assert status.operational is True


def test_client_property(mock_backend):
    """
    Test that the client property is an IonQClient instance.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
    """
    # Client will be lazily created.
    client = mock_backend.client
    assert isinstance(client, ionq_client.IonQClient)


def test_retrieve_job(mock_backend, requests_mock):
    """
    Test that retrieve job returns a valid job instance.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    path = mock_backend.client.make_path("jobs", "fake_job_id")
    requests_mock.get(path, json=conftest.dummy_job_response("fake_job_id"))
    job = mock_backend.retrieve_job("fake_job_id")
    assert isinstance(job, ionq_job.IonQJob)
    assert job.job_id() == "fake_job_id"


def test_retrieve_jobs(mock_backend, requests_mock):
    """
    Test that retrieve job returns a valid job instance.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    job_ids = [f"fake_job_{i}" for i in range(10)]
    for job_id in job_ids:
        path = mock_backend.client.make_path("jobs", job_id)
        requests_mock.get(path, json=conftest.dummy_job_response(job_id))
    jobs = mock_backend.retrieve_jobs(job_ids)

    # They're all jobs.
    assert all(isinstance(job, ionq_job.IonQJob) for job in jobs)

    # All IDs are accounted for
    assert sorted(job_ids) == sorted([job.job_id() for job in jobs])


@pytest.mark.parametrize(
    "creds,msg",
    [
        ({}, "Credentials `token` not present in provider."),
        ({"token": None}, "Credentials `token` may not be None!"),
        ({"token": "something"}, "Credentials `url` not present in provider."),
        (
            {"token": "something", "url": None},
            "Credentials `url` may not be None!",
        ),
    ],
)
def test_create_client_exceptions(mock_backend, creds, msg):
    """Test various exceptions that can be raised during client creation.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        creds (dict): A dictionary of bad credentials values.
        msg (str): An expected error for `creds`.
    """
    fake_provider = mock.MagicMock()
    fake_provider.credentials = creds
    provider_patch = mock.patch.object(mock_backend, "_provider", fake_provider)
    with provider_patch, pytest.raises(exceptions.IonQCredentialsError) as exc_info:
        mock_backend.create_client()

    assert str(exc_info.value.message) == msg


def test_run(mock_backend, requests_mock):
    """Test that the backend `run` submits a circuit and returns its job.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    path = mock_backend.client.make_path("jobs")
    dummy_response = conftest.dummy_job_response("fake_job")

    # Mock the call to submit:
    requests_mock.post(path, json=dummy_response, status_code=200)

    # Run a dummy circuit.
    qc = QuantumCircuit(1)
    qc.measure_all()
    job = mock_backend.run(qc)

    assert isinstance(job, ionq_job.IonQJob)
    assert job.job_id() == "fake_job"


def test_run_single_element_list(mock_backend, requests_mock):
    """Test that the backend `run` submits a circuit in a single-element list.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    path = mock_backend.client.make_path("jobs")
    dummy_response = conftest.dummy_job_response("fake_job")

    # Mock the call to submit:
    requests_mock.post(path, json=dummy_response, status_code=200)

    # Run a dummy circuit.
    qc = QuantumCircuit(1)
    qc.measure_all()
    job = mock_backend.run([qc])

    assert isinstance(job, ionq_job.IonQJob)
    assert job.job_id() == "fake_job"


def test_run_extras(mock_backend, requests_mock):
    """Test that the backend `run` accepts an arbitrary parameter dictionary.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    path = mock_backend.client.make_path("jobs")
    dummy_response = conftest.dummy_job_response("fake_job")

    # Mock the call to submit:
    requests_mock.post(path, json=dummy_response, status_code=200)

    # Run a dummy circuit.
    qc = QuantumCircuit(1, metadata={"experiment": "abc123"})
    qc.measure_all()
    job = mock_backend.run(
        qc,
        extra_query_params={
            "error_mitigation": {"debias": True},
        },
        extra_metadata={
            "iteration": "10",
        },
    )

    assert isinstance(job, ionq_job.IonQJob)
    assert job.job_id() == "fake_job"
    assert job.extra_query_params == {
        "error_mitigation": {"debias": True},
    }
    assert job.extra_metadata == {
        "iteration": "10",
    }
    assert job.circuit.metadata == {"experiment": "abc123"}


def test_warn_null_mappings(mock_backend, requests_mock):
    """Test that a circuit without measurements emits
    a warning.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    path = mock_backend.client.make_path("jobs")
    dummy_response = conftest.dummy_job_response("fake_job")

    # Mock the call to submit:
    requests_mock.post(path, json=dummy_response, status_code=200)

    # Create a circuit with no measurement gates
    qc = QuantumCircuit(1, 1)
    qc.h(0)

    with pytest.warns(UserWarning) as warninfo:
        mock_backend.run(qc)
    assert "Circuit is not measuring any qubits" in {str(w.message) for w in warninfo}


def test_multiexp_job(mock_backend, requests_mock):
    """Test that the backend `run` handles more than one circuit.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    path = mock_backend.client.make_path("jobs")
    dummy_response = conftest.dummy_job_response("fake_job")

    # Mock the call to submit:
    requests_mock.post(path, json=dummy_response, status_code=200)

    # Run a dummy multi-experiment job.
    qc1 = QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure(0, 0)
    qc2 = QuantumCircuit(1, 1)
    qc2.x(0)
    qc2.measure(0, 0)
    job = mock_backend.run([qc1, qc2])

    # Verify json payload
    assert len(job.circuit) == 2
    assert len(requests_mock.request_history) == 1
    request = requests_mock.request_history[0]
    assert request.method == "POST"
    assert request.url == path
    request_json = request.json()
    assert "qiskit_header" in request_json["metadata"]
    # delete the qiskit_header field
    del request_json["metadata"]["qiskit_header"]
    assert request_json == {
        "backend": "mock_backend",
        "shots": 1024,
        "name": f"{len(job.circuit)} circuits",
        "type": "ionq.multi-circuit.v1",
        "input": {
            "gateset": "qis",
            "qubits": 1,
            "circuits": [
                {
                    "name": qc1.name,
                    "circuit": [{"gate": "h", "targets": [0]}],
                    "registers": {"meas_mapped": [0]},
                },
                {
                    "name": qc2.name,
                    "circuit": [{"gate": "x", "targets": [0]}],
                    "registers": {"meas_mapped": [0]},
                },
            ],
        },
        "metadata": {
            "shots": "1024",
            "sampler_seed": "None",
        },
    }
