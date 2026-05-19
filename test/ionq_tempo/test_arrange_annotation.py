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

"""``@ionq.arrange`` annotation round-trip tests.

Covers:
  * Serializer dump/load symmetry for the standalone payload.
  * Full QASM 3 -> QuantumCircuit ingestion via qiskit.qasm3.loads.
  * Malformed-payload error path.
  * IonQJob.compiled_circuit() ingests an arrange annotation from the
    /circuits/qasm3 endpoint.
"""
# pylint: disable=redefined-outer-name

import pytest
from qiskit.qasm3 import loads as qasm3_loads

from qiskit_ionq import ionq_job
from qiskit_ionq.helpers import compress_to_metadata_string
from qiskit_ionq.ionq_annotations import (
    IONQ_ARRANGE_NAMESPACE,
    IonQArrangeAnnotation,
    IonQArrangeSerializer,
    ionq_annotation_handlers,
)


# ---------------------------------------------------------------------------
# Serializer-level round trip
# ---------------------------------------------------------------------------


def test_arrange_serializer_dump():
    """dump() produces a single-line JSON payload with the expected keys."""
    ann = IonQArrangeAnnotation(from_label="A12S", to_label="A23S", targets=(0, 1, 2))
    serialized = IonQArrangeSerializer().dump(ann)
    assert "\n" not in serialized
    assert '"from":"A12S"' in serialized
    assert '"to":"A23S"' in serialized
    assert '"targets":[0,1,2]' in serialized


def test_arrange_serializer_load():
    """load() reconstructs the annotation from a JSON payload."""
    payload = '{"from":"X","to":"Y","targets":[3,4]}'
    ann = IonQArrangeSerializer().load(IONQ_ARRANGE_NAMESPACE, payload)
    assert isinstance(ann, IonQArrangeAnnotation)
    assert ann.from_label == "X"
    assert ann.to_label == "Y"
    assert ann.targets == (3, 4)
    assert ann.namespace == IONQ_ARRANGE_NAMESPACE


def test_load_wrong_ns_skips():
    """Returning NotImplemented lets Qiskit try a different handler."""
    out = IonQArrangeSerializer().load("ionq.other", "{}")
    assert out is NotImplemented


def test_dump_wrong_type_skips():
    """dump() returns NotImplemented for annotations it doesn't own."""

    class DummyAnn:  # pylint: disable=missing-class-docstring,too-few-public-methods
        namespace = "other.namespace"

    out = IonQArrangeSerializer().dump(DummyAnn())
    assert out is NotImplemented


def test_load_malformed_raises():
    """A payload missing a required field surfaces as ValueError, not silently parsed wrong."""
    with pytest.raises(ValueError, match="Malformed"):
        IonQArrangeSerializer().load(IONQ_ARRANGE_NAMESPACE, '{"from":"X"}')


# ---------------------------------------------------------------------------
# QASM 3 -> QuantumCircuit ingestion
# ---------------------------------------------------------------------------


_ARRANGE_QASM = """OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;

h q[0];
@ionq.arrange {"from":"A12S","to":"A23S","targets":[0,1]}
box {
}
cx q[0], q[1];
"""


def test_qasm3_loads_attaches_ann():
    """The annotation handler attaches the IonQArrangeAnnotation to the box op."""
    qc = qasm3_loads(_ARRANGE_QASM, annotation_handlers=ionq_annotation_handlers())
    box_insts = [inst for inst in qc.data if inst.operation.name == "box"]
    assert len(box_insts) == 1
    annotations = box_insts[0].operation.annotations
    assert len(annotations) == 1
    ann = annotations[0]
    assert isinstance(ann, IonQArrangeAnnotation)
    assert ann.from_label == "A12S"
    assert ann.to_label == "A23S"
    assert ann.targets == (0, 1)


# ---------------------------------------------------------------------------
# IonQJob.compiled_circuit() integration
# ---------------------------------------------------------------------------


def _dry_run_v2_response(job_id: str) -> dict:
    """A completed dry-run job that produced a v2-style compiled circuit."""
    return {
        "id": job_id,
        "type": "ionq.circuit.v2",
        "status": "completed",
        "backend": "qpu.tempo-1",
        "dry_run": True,
        "metadata": {
            "shots": "100",
            "qiskit_header": compress_to_metadata_string(
                {
                    "n_qubits": 2,
                    "memory_slots": 0,
                    "name": "dry",
                    "qreg_sizes": [["q", 2]],
                    "qubit_labels": [["q", 0], ["q", 1]],
                    "creg_sizes": [],
                    "clbit_labels": [],
                    "global_phase": 0,
                }
            ),
        },
    }


@pytest.fixture
def arrange_job(provider, requests_mock):
    """A dry-run Tempo job whose compiled QASM 3 contains @ionq.arrange."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="qis")
    client = backend._create_client()  # pylint: disable=protected-access
    job_id = "arrange_demo"
    requests_mock.get(
        client.make_path("jobs", job_id),
        json=_dry_run_v2_response(job_id),
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "circuits", "qasm3"),
        json=_ARRANGE_QASM,
    )
    return ionq_job.IonQJob(backend, job_id, client)


def test_compiled_circuit_arrange(arrange_job):
    """compiled_circuit() routes the annotation handler through qasm3.loads."""
    qc = arrange_job.compiled_circuit()
    box_insts = [inst for inst in qc.data if inst.operation.name == "box"]
    assert len(box_insts) == 1
    ann = box_insts[0].operation.annotations[0]
    assert isinstance(ann, IonQArrangeAnnotation)
    assert ann.from_label == "A12S"
    assert ann.to_label == "A23S"
    assert ann.targets == (0, 1)
