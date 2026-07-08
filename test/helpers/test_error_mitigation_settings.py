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

# Copyright 2024 IonQ, Inc. (www.ionq.com)
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

"""Tests for error mitigation settings serialization and result aggregation."""

import json

import pytest

from qiskit import QuantumCircuit

from qiskit_ionq.helpers import qiskit_to_ionq
from qiskit_ionq.constants import AggregationMethod
from qiskit_ionq import ionq_job
from ..utils import dummy_job_response


def _simple_circuit():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _extract_em_settings(ionq_json):
    """Extract settings.error_mitigation from a serialized job payload."""
    return json.loads(ionq_json).get("settings", {}).get("error_mitigation", {})


def _setup_job(mock_backend, requests_mock, results_path_suffix=""):
    """Set up a completed mock job with an optional results URL suffix."""
    job_id = "test_id"
    client = mock_backend.client
    path = client.make_path("jobs", job_id)
    results_base = client.make_path("jobs", job_id, "results", "probabilities")
    requests_mock.get(path, status_code=200, json=dummy_job_response(job_id))
    requests_mock.get(
        results_base + results_path_suffix,
        status_code=200,
        json={"0": 0.5, "1": 0.5},
    )
    return ionq_job.IonQJob(mock_backend, job_id)


# ---------------------------------------------------------------------------
# Flat kwargs: debiasing=
# ---------------------------------------------------------------------------


def test_debiasing_true(simulator_backend):
    """debiasing=True serializes as {"debiasing": True}."""
    args = {"shots": 10, "debiasing": True}
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {"debiasing": True}


def test_debiasing_false(simulator_backend):
    """debiasing=False serializes as {"debiasing": False}."""
    args = {"shots": 10, "debiasing": False}
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {"debiasing": False}


def test_debiasing_none_produces_no_em_block(simulator_backend):
    """debiasing=None produces no error_mitigation block."""
    args = {"shots": 10, "debiasing": None}
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {}


# ---------------------------------------------------------------------------
# Flat kwargs: symmetry_verification=
# ---------------------------------------------------------------------------


def test_sv_true(simulator_backend):
    """symmetry_verification=True serializes correctly."""
    args = {"shots": 10, "symmetry_verification": True}
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {"symmetry_verification": True}


def test_sv_false(simulator_backend):
    """symmetry_verification=False serializes correctly."""
    args = {"shots": 10, "symmetry_verification": False}
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {"symmetry_verification": False}


def test_sv_none_produces_no_em_block(simulator_backend):
    """symmetry_verification=None produces no error_mitigation block."""
    args = {"shots": 10, "symmetry_verification": None}
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {}


def test_both_flat_kwargs(simulator_backend):
    """Both flat kwargs serialize together."""
    args = {"shots": 10, "debiasing": False, "symmetry_verification": False}
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {"debiasing": False, "symmetry_verification": False}


# ---------------------------------------------------------------------------
# job_settings escape hatch
# ---------------------------------------------------------------------------


def test_flat_kwarg_merges_with_job_settings(simulator_backend):
    """Flat kwarg merges with an existing job_settings EM block."""
    args = {
        "shots": 10,
        "job_settings": {"error_mitigation": {"symmetry_verification": True}},
        "debiasing": False,
    }
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {"symmetry_verification": True, "debiasing": False}


# ---------------------------------------------------------------------------
# No EM args
# ---------------------------------------------------------------------------


def test_no_em_args_produces_no_em_block(simulator_backend):
    """No EM kwargs produces no error_mitigation block in the payload."""
    args = {"shots": 10}
    em_block = _extract_em_settings(
        qiskit_to_ionq(_simple_circuit(), simulator_backend, passed_args=args)
    )
    assert em_block == {}


# ---------------------------------------------------------------------------
# result() aggregation kwarg
# ---------------------------------------------------------------------------


def test_default_no_aggregation_param(mock_backend, requests_mock):
    """result() with no aggregation kwarg sends no aggregation query param."""
    job = _setup_job(mock_backend, requests_mock)
    assert job.result() is not None
    assert "aggregation" not in requests_mock.last_request.qs


def test_aggregation_voting_string(mock_backend, requests_mock):
    """aggregation='voting' sends ?aggregation=voting."""
    job = _setup_job(mock_backend, requests_mock, "?aggregation=voting")
    assert job.result(aggregation="voting") is not None


def test_aggregation_dnl_string(mock_backend, requests_mock):
    """aggregation='dnl' sends ?aggregation=dnl."""
    job = _setup_job(mock_backend, requests_mock, "?aggregation=dnl")
    assert job.result(aggregation="dnl") is not None


def test_aggregation_enum(mock_backend, requests_mock):
    """AggregationMethod.VOTING enum sends ?aggregation=voting."""
    job = _setup_job(mock_backend, requests_mock, "?aggregation=voting")
    assert job.result(aggregation=AggregationMethod.VOTING) is not None


def test_aggregation_dnl_enum(mock_backend, requests_mock):
    """AggregationMethod.DNL enum sends ?aggregation=dnl."""
    job = _setup_job(mock_backend, requests_mock, "?aggregation=dnl")
    assert job.result(aggregation=AggregationMethod.DNL) is not None


def test_sharpen_true_deprecated_maps_to_voting(mock_backend, requests_mock):
    """sharpen=True emits DeprecationWarning and maps to aggregation='voting'."""
    job = _setup_job(mock_backend, requests_mock, "?aggregation=voting")
    with pytest.warns(DeprecationWarning, match="sharpen is deprecated"):
        result = job.result(sharpen=True)
    assert result is not None


def test_sharpen_false_deprecated_no_aggregation(mock_backend, requests_mock):
    """sharpen=False also emits DeprecationWarning but sends no aggregation param."""
    job = _setup_job(mock_backend, requests_mock)
    with pytest.warns(DeprecationWarning, match="sharpen is deprecated"):
        result = job.result(sharpen=False)
    assert result is not None
    assert "aggregation" not in requests_mock.last_request.qs


def test_client_get_results_sharpen_positional_back_compat(mock_backend, requests_mock):
    """IonQClient.get_results still accepts sharpen positionally, mapping True
    to aggregation='voting' with a DeprecationWarning."""
    client = mock_backend.client
    job_id = "test_id"
    # Relative URL as returned by the v0.4 API in a job's results block.
    results_url = f"/v0.4/jobs/{job_id}/results/probabilities"
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities")
        + "?aggregation=voting",
        status_code=200,
        json={"0": 0.5, "1": 0.5},
    )
    with pytest.warns(DeprecationWarning, match="sharpen is deprecated"):
        result = client.get_results(results_url, True)
    assert result == {"0": 0.5, "1": 0.5}


# ---------------------------------------------------------------------------
# Integration: passing EM kwargs directly into backend.run()
# ---------------------------------------------------------------------------


def _mock_submit(backend, requests_mock):
    """Register a POST /jobs mock and return the path so callers can inspect requests."""
    path = backend.client.make_path("jobs")
    requests_mock.post(path, json=dummy_job_response("fake_job"), status_code=200)
    return path


def _posted_em_settings(requests_mock):
    """Return the error_mitigation block from the most-recently POSTed job payload."""
    body = json.loads(requests_mock.last_request.body)
    return body.get("settings", {}).get("error_mitigation", {})


def test_run_with_debiasing_true(mock_backend, requests_mock):
    """backend.run(circuit, debiasing=True) sends {"debiasing": true} in settings."""
    _mock_submit(mock_backend, requests_mock)
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    mock_backend.run(qc, shots=10, debiasing=True)
    assert _posted_em_settings(requests_mock) == {"debiasing": True}
