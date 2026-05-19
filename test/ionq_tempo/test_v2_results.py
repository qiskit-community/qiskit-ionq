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

"""Result-side tests for the ``ionq.circuit.v2`` (Tempo) payload.

Covers:
  * ``IonQJob`` detects the v2 ``type`` and stores per-format result URLs.
  * ``IonQResult.get_counts()`` aggregates ``output_all`` for backward compat.
  * ``IonQResult.probabilities_by_register()`` surfaces per-register data.
  * ``IonQJob.shots()`` / ``IonQJob.histogram()`` lazy-fetch the new endpoints.
  * ``IonQResult.get_leakage()`` surfaces subspace-leakage flags.
"""
# pylint: disable=redefined-outer-name

import pytest

from qiskit_ionq import ionq_job
from qiskit_ionq.exceptions import IonQClientError, IonQJobError
from qiskit_ionq.helpers import compress_to_metadata_string


# ---------------------------------------------------------------------------
# Fixtures: a v2 job response and the three results endpoints.
# ---------------------------------------------------------------------------


_V2_JOB_ID = "v2_job_id"


def _v2_qiskit_header(name: str = "tempo_demo") -> str:
    return compress_to_metadata_string(
        {
            "n_qubits": 1,
            "memory_slots": 2,
            "name": name,
            "qreg_sizes": [["q", 1]],
            "qubit_labels": [["q", 0]],
            "creg_sizes": [["mid", 1], ["output_all", 1]],
            "clbit_labels": [["mid", 0], ["output_all", 0]],
            "global_phase": 0,
        }
    )


def _v2_job_response(job_id: str = _V2_JOB_ID) -> dict:
    """Job-record payload as the API returns for a completed v2 job."""
    return {
        "id": job_id,
        "type": "ionq.circuit.v2",
        "status": "completed",
        "backend": "qpu.tempo-1",
        "results": {
            "shots": {"url": f"/v0.4/jobs/{job_id}/results/shots"},
            "probabilities": {"url": f"/v0.4/jobs/{job_id}/results/probabilities"},
            "histogram": {"url": f"/v0.4/jobs/{job_id}/results/histogram"},
        },
        "metadata": {
            "shots": "1000",
            "qiskit_header": _v2_qiskit_header(),
        },
        "execution_duration_ms": 250,
        "name": "tempo_demo",
    }


_V2_PROBABILITIES = {
    "probabilities": {
        "mid": {"0": 0.49, "1": 0.51},
        "output_all": {"0": 0.50, "1": 0.50},
    }
}


_V2_SHOTS = {
    "shots": [
        {"mid": [[0]], "output_all": [[0]]},
        {"mid": [[1]], "output_all": [[1]]},
        {"mid": [[0]], "output_all": [[1]]},
    ]
}


_V2_HISTOGRAM = {
    "histogram": {
        "mid": {"0": 490, "1": 510},
        "output_all": {"0": 500, "1": 500},
    }
}


@pytest.fixture
def tempo_job(provider, requests_mock):
    """An IonQJob attached to a completed v2 job, with all endpoints mocked."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="qis")
    client = backend._create_client()  # pylint: disable=protected-access
    requests_mock.get(
        client.make_path("jobs", _V2_JOB_ID),
        json=_v2_job_response(_V2_JOB_ID),
    )
    requests_mock.get(
        client.make_path("jobs", _V2_JOB_ID, "results", "probabilities"),
        json=_V2_PROBABILITIES,
    )
    requests_mock.get(
        client.make_path("jobs", _V2_JOB_ID, "results", "shots"),
        json=_V2_SHOTS,
    )
    requests_mock.get(
        client.make_path("jobs", _V2_JOB_ID, "results", "histogram"),
        json=_V2_HISTOGRAM,
    )
    return ionq_job.IonQJob(backend, _V2_JOB_ID, client)


# ---------------------------------------------------------------------------
# v2 detection
# ---------------------------------------------------------------------------


def test_v2_job_detection(tempo_job):
    """The job recognises ``type: ionq.circuit.v2`` and stores all three URLs."""
    assert tempo_job._is_v2 is True  # pylint: disable=protected-access
    assert tempo_job._results_url.endswith("/results/probabilities")  # pylint: disable=protected-access
    assert tempo_job._shots_url.endswith("/results/shots")  # pylint: disable=protected-access
    assert tempo_job._histogram_url.endswith("/results/histogram")  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Probability decoding
# ---------------------------------------------------------------------------


def test_v2_counts_from_output_all(tempo_job):
    """get_counts() reflects the output_all register (the v1-compatible default)."""
    result = tempo_job.result()
    counts = result.get_counts()
    # output_all = {"0": 0.5, "1": 0.5}, shots=1000 -> 500 of each
    assert counts == {"0": 500, "1": 500}


def test_v2_probs_by_register(tempo_job):
    """probabilities_by_register() exposes each declared classical register."""
    result = tempo_job.result()
    per_reg = result.probabilities_by_register()
    assert set(per_reg.keys()) == {"mid", "output_all"}
    assert per_reg["mid"] == {"0": 0.49, "1": 0.51}
    assert per_reg["output_all"] == {"0": 0.50, "1": 0.50}


def test_v1_probs_by_reg_raises(formatted_result):
    """V1 results don't have per-register data; the accessor raises clearly."""
    with pytest.raises(IonQJobError, match="per-register"):
        formatted_result.probabilities_by_register()


# ---------------------------------------------------------------------------
# Lazy /shots and /histogram fetches
# ---------------------------------------------------------------------------


def test_v2_shots_endpoint(tempo_job):
    """job.shots() round-trips the shot list unmodified."""
    shots = tempo_job.shots()
    assert len(shots) == 3
    assert shots[0] == {"mid": [[0]], "output_all": [[0]]}
    assert shots[2] == {"mid": [[0]], "output_all": [[1]]}


def test_v2_histogram_endpoint(tempo_job):
    """job.histogram() returns the per-register count map."""
    hist = tempo_job.histogram()
    assert hist["mid"] == {"0": 490, "1": 510}
    assert hist["output_all"] == {"0": 500, "1": 500}


def test_v1_shots_unavailable(provider, requests_mock):
    """Non-v2 backends have no shots URL; the accessor raises a clear error."""
    backend = provider.get_backend("ionq_qpu.aria-1")
    client = backend._create_client()  # pylint: disable=protected-access
    job_id = "v1_job"
    requests_mock.get(
        client.make_path("jobs", job_id),
        json={
            "id": job_id,
            "type": "ionq.circuit.v1",
            "status": "completed",
            "backend": "qpu.aria-1",
            "results": {
                "probabilities": {"url": f"/v0.4/jobs/{job_id}/results/probabilities"}
            },
            "metadata": {"shots": "1000", "qiskit_header": _v2_qiskit_header()},
            "execution_duration_ms": 100,
        },
    )
    job = ionq_job.IonQJob(backend, job_id, client)
    with pytest.raises(IonQClientError, match="Tempo-class"):
        job.shots()
    with pytest.raises(IonQClientError, match="Tempo-class"):
        job.histogram()


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_v2_get_leakage_absent(tempo_job):
    """When include_leakage was not requested, get_leakage() returns None."""
    result = tempo_job.result()
    assert result.get_leakage() is None


def test_v2_get_leakage_present(provider, requests_mock):
    """include_leakage=true: leakage bits ride under output.error_mitigation."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="qis")
    client = backend._create_client()  # pylint: disable=protected-access
    job_id = "v2_leakage_job"
    requests_mock.get(
        client.make_path("jobs", job_id),
        json=_v2_job_response(job_id),
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities"),
        json={
            "probabilities": {"output_all": {"0": 1.0}},
            "output": {
                "error_mitigation": {
                    "leakage": [[0, 0], [0, 1], [0, 0]],
                }
            },
        },
    )
    job = ionq_job.IonQJob(backend, job_id, client)
    leakage = job.result().get_leakage()
    assert leakage == [[0, 0], [0, 1], [0, 0]]
