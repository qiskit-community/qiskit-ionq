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
from collections import Counter

import pytest
from qiskit import QuantumCircuit, transpile

from qiskit_ionq import exceptions, ionq_client, ionq_job
from qiskit_ionq.helpers import get_user_agent

from .. import conftest


def test_simulator_status_is_true(mock_backend):
    """Simulator ``status()`` returns ``True`` (short-circuits before HTTP)."""
    assert mock_backend.status() is True


def test_qpu_status_with_real_cal(qpu_backend, requests_mock):
    """Real-QPU ``Characterization`` payloads omit ``status`` per the v0.4
    spec, so ``Characterization.status`` must fall back, not ``KeyError``.
    """
    client = qpu_backend.client
    url = client.make_path(
        "backends", qpu_backend._api_backend_name, "characterizations"
    )
    requests_mock.get(
        url,
        json={
            "characterizations": [
                {
                    "id": "abc-123",
                    "backend": qpu_backend._api_backend_name,
                    "date": "2026-01-01T00:00:00Z",
                    "qubits": 25,
                    "connectivity": [[0, 1], [0, 2]],
                    "fidelity": {"spam": {"median": 0.99}},
                    "timing": {},
                }
            ],
            "pages": 0,
        },
    )
    assert qpu_backend.status() is True


def test_get_cal_data_no_chars(qpu_backend, requests_mock):
    """``{"characterizations": null}`` (seen on backends without data like
    ``qpu.qpu``) must yield ``[]``, not ``TypeError``. The convenience
    wrapper returns ``None`` in the same situation. Backend.status() must
    surface that as ``False``, not crash.
    """
    client = qpu_backend.client
    name = qpu_backend._api_backend_name
    url = client.make_path("backends", name, "characterizations")
    requests_mock.get(url, json={"characterizations": None, "pages": 0})
    assert client.get_calibration_data(name, limit=1) == []
    assert client.get_latest_calibration(name) is None
    assert qpu_backend.status() is False


def test_get_cal_data_always_list(qpu_backend, requests_mock):
    """``get_calibration_data`` returns a list regardless of ``limit``;
    ``get_latest_calibration`` is the way to ask for a single entry.
    """
    client = qpu_backend.client
    name = qpu_backend._api_backend_name
    url = client.make_path("backends", name, "characterizations")

    def _cal(uid):
        return {
            "id": uid,
            "backend": name,
            "date": "2026-01-01T00:00:00Z",
            "qubits": 25,
            "connectivity": [[0, 1]],
            "fidelity": {},
            "timing": {},
        }

    requests_mock.get(
        url,
        json={"characterizations": [_cal("a"), _cal("b")], "pages": 1},
    )
    chars = client.get_calibration_data(name, limit=2)
    assert isinstance(chars, list) and len(chars) == 2
    latest = client.get_latest_calibration(name)
    assert latest is not None and latest.id == "a"


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
        mock_backend._create_client()

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
                },
                {
                    "name": qc2.name,
                    "circuit": [{"gate": "x", "targets": [0]}],
                },
            ],
        },
        "metadata": {
            "shots": "1024",
            "sampler_seed": "None",
            "user_agent": get_user_agent(),
        },
    }


def test_backend_memory(mock_backend, requests_mock):
    """Test that memory is handled correctly.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """

    job_id = "mem_job"
    probabilities = {"0": 0.4, "1": 0.1, "2": 0.1, "3": 0.4}
    client = mock_backend.client
    resp = conftest.dummy_job_response(job_id)

    requests_mock.get(client.make_path("jobs", job_id), status_code=200, json=resp)
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities"),
        status_code=200,
        json=probabilities,
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "shots"),
        status_code=200,
        # The IonQ /results/shots endpoint returns a list of decimal-encoded
        # outcome integers (same convention as histogram keys), NOT preformatted
        # bitstrings. e.g. "3" means binary 11 on 2 qubits.
        json=["0", "0", "0", "0", "1", "2", "3", "3", "3", "3"],
    )

    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": True, "shots": 1024, "sampler_seed": None},
    )
    memory = Counter(job.get_memory())
    assert memory == Counter({"00": 4, "11": 4, "01": 1, "10": 1})


def test_run_dry_run_sends_flag(mock_backend, requests_mock):
    """`backend.run(qc, dry_run=True)` must put `dry_run: true` at the top
    level of the request body. Compilation-as-a-service uses this to compile
    the circuit without executing it on the QPU."""
    path = mock_backend.client.make_path("jobs")
    requests_mock.post(
        path, json=conftest.dummy_job_response("fake_job"), status_code=200
    )

    qc = QuantumCircuit(1)
    qc.measure_all()
    mock_backend.run(qc, dry_run=True)

    assert len(requests_mock.request_history) == 1
    request_json = requests_mock.request_history[0].json()
    assert request_json["dry_run"] is True


def test_run_no_dry_run_omits_flag(mock_backend, requests_mock):
    """When `dry_run` is not supplied, the field must NOT be in the request
    body at all (so the API behaves identically to pre-1.x clients)."""
    path = mock_backend.client.make_path("jobs")
    requests_mock.post(
        path, json=conftest.dummy_job_response("fake_job"), status_code=200
    )

    qc = QuantumCircuit(1)
    qc.measure_all()
    mock_backend.run(qc)

    request_json = requests_mock.request_history[0].json()
    assert "dry_run" not in request_json


def test_dry_run_via_extra_params(mock_backend, requests_mock):
    """Back-compat: the pre-existing
    `extra_query_params={"dry_run": True}` workaround must keep working
    after the first-class kwarg is added."""
    path = mock_backend.client.make_path("jobs")
    requests_mock.post(
        path, json=conftest.dummy_job_response("fake_job"), status_code=200
    )

    qc = QuantumCircuit(1)
    qc.measure_all()
    mock_backend.run(qc, extra_query_params={"dry_run": True})

    request_json = requests_mock.request_history[0].json()
    assert request_json["dry_run"] is True


def test_native_sim_target_noise(provider):
    """Test that the simulator native target 2q family follows the noise model
    (the target advertises standard-gate proxies: rzz for zz, rxx for ms).

    Args:
        provider (IonQProvider): A test IonQProvider.
    """
    sim_backend = provider.get_backend("ionq_simulator", gateset="native")

    target_ops = [op.name for op in sim_backend.target.operations]
    assert "rxx" in target_ops
    assert "rzz" not in target_ops

    sim_backend.set_options(noise_model="forte-1")
    target_ops = [op.name for op in sim_backend.target.operations]
    assert "rzz" in target_ops
    assert "rxx" not in target_ops

    sim_backend.set_options(noise_model="aria-1")
    target_ops = [op.name for op in sim_backend.target.operations]
    assert "rxx" in target_ops
    assert "rzz" not in target_ops


def test_forte_rzz_transpiles_to_zz(provider):
    """Test that rzz gates transpile to zz (not ms) with Forte noise model.

    Regression test for issue #210.

    Args:
        provider (IonQProvider): A test IonQProvider.
    """
    native_simulator = provider.get_backend("ionq_simulator", gateset="native")
    native_simulator.set_options(noise_model="forte-1")

    qc = QuantumCircuit(2)
    qc.rzz(1, 0, 1)

    transpiled_qc = transpile(qc, backend=native_simulator, optimization_level=3)
    ops = transpiled_qc.count_ops()

    assert "zz" in ops
    assert "ms" not in ops
