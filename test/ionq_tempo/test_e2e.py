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

"""End-to-end Tempo flow against mocked v2 endpoints.

Mocks the full submission -> poll -> retrieve path through
``requests_mock`` and asserts that the qiskit-ionq client speaks v2
correctly on every hop: ``ionq.circuit.v2`` submission with a QASM 3
``input`` string, then the three new results endpoints, then the
compiled-circuit readback path with an ``@ionq.arrange`` annotation.
"""
# pylint: disable=redefined-outer-name

import json

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qiskit_ionq.helpers import compress_to_metadata_string
from qiskit_ionq.ionq_annotations import IonQArrangeAnnotation


_JOB_ID = "tempo-e2e-id"


def _v2_completed_response() -> dict:
    """Mock /v0.4/jobs/{id} payload that matches the v2 spec."""
    qiskit_header = compress_to_metadata_string(
        {
            "n_qubits": 1,
            "memory_slots": 2,
            "name": "mcm_demo",
            "qreg_sizes": [["q", 1]],
            "qubit_labels": [["q", 0]],
            "creg_sizes": [["mid", 1], ["output_all", 1]],
            "clbit_labels": [["mid", 0], ["output_all", 0]],
            "global_phase": 0,
        }
    )
    return {
        "id": _JOB_ID,
        "type": "ionq.circuit.v2",
        "status": "completed",
        "backend": "qpu.tempo-1",
        "results": {
            "shots": {"url": f"/v0.4/jobs/{_JOB_ID}/results/shots"},
            "probabilities": {"url": f"/v0.4/jobs/{_JOB_ID}/results/probabilities"},
            "histogram": {"url": f"/v0.4/jobs/{_JOB_ID}/results/histogram"},
        },
        "metadata": {
            "shots": "1000",
            "qiskit_header": qiskit_header,
        },
        "execution_duration_ms": 250,
        "name": "mcm_demo",
    }


_PROBABILITIES = {
    "probabilities": {
        "mid": {"0": 0.49, "1": 0.51},
        "output_all": {"0": 0.50, "1": 0.50},
    },
    "output": {
        "error_mitigation": {
            "leakage": [[0], [0], [0], [1]],
        }
    },
}


_SHOTS = {
    "shots": [
        {"mid": [[0]], "output_all": [[0]]},
        {"mid": [[1]], "output_all": [[1]]},
        {"mid": [[0]], "output_all": [[1]]},
    ]
}


_HISTOGRAM = {
    "histogram": {
        "mid": {"0": 490, "1": 510},
        "output_all": {"0": 500, "1": 500},
    }
}


# A simplified compiled-circuit fixture. The real Cypress MIR->QASM 3
# transpiler emits ``gate gpi2(phi) q { ... }`` declarations alongside
# the native-gate invocations; for this test we exercise the annotation
# handler with stdlib gates so the body parses without IonQ-specific
# gate definitions.
_COMPILED_QASM3 = """OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
bit[1] mid;
bit[2] output_all;

h q[0];
@ionq.arrange {"from":"A12S","to":"A23S","targets":[0,1]}
box {
}
cx q[0], q[1];
mid[0] = measure q[0];
output_all[0] = measure q[0];
output_all[1] = measure q[1];
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mcm_circuit():
    """Mid-circuit-measurement circuit Tempo is meant to accept."""
    qr = QuantumRegister(1, "q")
    mid = ClassicalRegister(1, "mid")
    out = ClassicalRegister(1, "output_all")
    qc = QuantumCircuit(qr, mid, out, name="mcm_demo")
    qc.h(0)
    qc.measure(0, mid[0])
    qc.x(0)
    qc.measure(0, out[0])
    return qc


@pytest.fixture
def tempo_backend(provider):
    """Tempo backend with the QIS gateset (the typical Tempo user path)."""
    return provider.get_backend("ionq_qpu.tempo-1", gateset="qis")


@pytest.fixture
def mocked_v2(tempo_backend, requests_mock):
    """Wire up the entire v2 submission + result + compiled-circuit path."""
    client = tempo_backend._create_client()  # pylint: disable=protected-access
    # POST /jobs returns just the id+status (the API does not echo the payload).
    requests_mock.post(
        client.make_path("jobs"),
        json={"id": _JOB_ID, "status": "submitted"},
    )
    requests_mock.get(
        client.make_path("jobs", _JOB_ID),
        json=_v2_completed_response(),
    )
    requests_mock.get(
        client.make_path("jobs", _JOB_ID, "results", "probabilities"),
        json=_PROBABILITIES,
    )
    requests_mock.get(
        client.make_path("jobs", _JOB_ID, "results", "shots"),
        json=_SHOTS,
    )
    requests_mock.get(
        client.make_path("jobs", _JOB_ID, "results", "histogram"),
        json=_HISTOGRAM,
    )
    requests_mock.get(
        client.make_path("jobs", _JOB_ID, "circuits", "qasm3"),
        json=_COMPILED_QASM3,
    )
    return requests_mock


# ---------------------------------------------------------------------------
# Submission shape
# ---------------------------------------------------------------------------


def test_submission_is_v2(tempo_backend, mcm_circuit, mocked_v2):
    """The POST body is ionq.circuit.v2 with an OpenQASM 3 ``input`` string."""
    tempo_backend.run(mcm_circuit, shots=1000)
    post_req = mocked_v2.request_history[0]
    body = json.loads(post_req.text)
    assert body["type"] == "ionq.circuit.v2"
    assert body["backend"] == "qpu.tempo-1"
    assert isinstance(body["input"], str)
    assert "OPENQASM 3" in body["input"]
    # Mid-circuit measure survives serialisation.
    assert body["input"].count("measure q[0]") == 2


# ---------------------------------------------------------------------------
# Result path
# ---------------------------------------------------------------------------


def test_result_get_counts(tempo_backend, mcm_circuit, mocked_v2):  # pylint: disable=unused-argument
    """get_counts() reflects output_all (the v1-compatible default register)."""
    job = tempo_backend.run(mcm_circuit, shots=1000)
    counts = job.result().get_counts()
    assert counts == {"0": 500, "1": 500}


def test_result_per_register(tempo_backend, mcm_circuit, mocked_v2):  # pylint: disable=unused-argument
    """probabilities_by_register() exposes both 'mid' and 'output_all'."""
    job = tempo_backend.run(mcm_circuit, shots=1000)
    per_reg = job.result().probabilities_by_register()
    assert set(per_reg.keys()) == {"mid", "output_all"}
    assert per_reg["mid"]["1"] == pytest.approx(0.51)


def test_result_leakage(tempo_backend, mcm_circuit, mocked_v2):  # pylint: disable=unused-argument
    """get_leakage() returns the per-shot leakage bits."""
    job = tempo_backend.run(mcm_circuit, shots=1000)
    leakage = job.result().get_leakage()
    assert leakage == [[0], [0], [0], [1]]


def test_shots_endpoint(tempo_backend, mcm_circuit, mocked_v2):  # pylint: disable=unused-argument
    """job.shots() returns the shot-wise list keyed by register name."""
    job = tempo_backend.run(mcm_circuit, shots=1000)
    shots = job.shots()
    assert [s["mid"] for s in shots] == [[[0]], [[1]], [[0]]]


def test_histogram_endpoint(tempo_backend, mcm_circuit, mocked_v2):  # pylint: disable=unused-argument
    """job.histogram() returns aggregated counts per register."""
    job = tempo_backend.run(mcm_circuit, shots=1000)
    hist = job.histogram()
    assert hist["mid"] == {"0": 490, "1": 510}


# ---------------------------------------------------------------------------
# Compiled circuit + ARRANGE
# ---------------------------------------------------------------------------


def test_compiled_has_arrange(tempo_backend, mcm_circuit, mocked_v2):  # pylint: disable=unused-argument
    """compiled_circuit() parses the @ionq.arrange annotation into the BoxOp."""
    job = tempo_backend.run(mcm_circuit, shots=1000)
    compiled = job.compiled_circuit()
    box_insts = [inst for inst in compiled.data if inst.operation.name == "box"]
    assert len(box_insts) == 1
    ann = box_insts[0].operation.annotations[0]
    assert isinstance(ann, IonQArrangeAnnotation)
    assert ann.from_label == "A12S"
    assert ann.targets == (0, 1)
