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

"""``qiskit_to_ionq`` -> ``ionq.circuit.v2`` payload tests.

The v2 schema is what Tempo-class backends consume:
``{"type": "ionq.circuit.v2", "input": "<openqasm 3 string>", ...}``.
These tests pin the wire shape and the dispatcher in
:func:`qiskit_ionq.helpers.qiskit_to_ionq`.
"""
# pylint: disable=redefined-outer-name

import json

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qiskit_ionq import GPI2Gate, ZZGate
from qiskit_ionq.exceptions import IonQGateError, IonQJobError
from qiskit_ionq.helpers import (
    decompress_metadata_string,
    is_v2_backend,
    qiskit_to_ionq,
)


# ---------------------------------------------------------------------------
# Backend-family dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ionq_qpu.tempo-1", True),
        ("qpu.tempo-1", True),
        ("tempo", True),
        ("ionq_qpu.tempo-2", True),
        ("ionq_qpu.aria-1", False),
        ("ionq_qpu.forte-1", False),
        ("ionq_qpu.forte-enterprise-1", False),
        ("ionq_simulator", False),
        ("simulator", False),
        ("", False),
        (None, False),
    ],
)
def test_is_v2_backend(name, expected):
    """The v2-payload dispatcher matches any tempo-flavoured name and only those."""
    assert is_v2_backend(name) is expected


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


@pytest.fixture
def tempo_backend(provider):
    """Native-gateset Tempo backend (the typical user path for v2)."""
    return provider.get_backend("ionq_qpu.tempo-1", gateset="native")


@pytest.fixture
def tempo_qis_backend(provider):
    """QIS-gateset Tempo backend (server-side compiler will resynthesize)."""
    return provider.get_backend("ionq_qpu.tempo-1", gateset="qis")


def test_v2_type_and_backend(tempo_qis_backend):
    """Tempo submission emits ``type=ionq.circuit.v2`` and the stripped backend name."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    payload = json.loads(
        qiskit_to_ionq(qc, tempo_qis_backend, passed_args={"shots": 50})
    )
    assert payload["type"] == "ionq.circuit.v2"
    assert payload["backend"] == "qpu.tempo-1"
    assert payload["shots"] == 50


def test_v2_input_is_qasm3_string(tempo_qis_backend):
    """``input`` carries an OpenQASM 3 source string (not the v1 dict)."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    payload = json.loads(
        qiskit_to_ionq(qc, tempo_qis_backend, passed_args={"shots": 50})
    )
    assert isinstance(payload["input"], str)
    assert payload["input"].startswith("OPENQASM 3")
    assert "h q[0]" in payload["input"]
    assert "measure q[0]" in payload["input"]


def test_v2_preserves_mid_measure(tempo_qis_backend):
    """Mid-circuit measure must survive serialization (no exception raised, kept in QASM)."""
    qr = QuantumRegister(1, "q")
    mid = ClassicalRegister(1, "mid")
    out = ClassicalRegister(1, "out")
    qc = QuantumCircuit(qr, mid, out)
    qc.h(0)
    qc.measure(0, mid[0])
    qc.x(0)
    qc.measure(0, out[0])

    payload = json.loads(
        qiskit_to_ionq(qc, tempo_qis_backend, passed_args={"shots": 50})
    )
    qasm = payload["input"]
    # Both measurements present, ordered, and bound to their named registers.
    assert qasm.count("measure q[0]") == 2
    assert "mid[0] = measure q[0]" in qasm
    assert "out[0] = measure q[0]" in qasm


def test_v2_named_registers(tempo_qis_backend):
    """Classical-register names declared by the user appear verbatim in the QASM."""
    qr = QuantumRegister(2, "q")
    rega = ClassicalRegister(1, "alpha")
    regb = ClassicalRegister(2, "beta")
    qc = QuantumCircuit(qr, rega, regb)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, rega[0])
    qc.measure([0, 1], [regb[0], regb[1]])

    payload = json.loads(
        qiskit_to_ionq(qc, tempo_qis_backend, passed_args={"shots": 100})
    )
    qasm = payload["input"]
    assert "bit[1] alpha" in qasm
    assert "bit[2] beta" in qasm


def test_v2_metadata_round_trip(tempo_qis_backend):
    """The compressed qiskit_header survives so result decoding can reconstruct registers."""
    qr = QuantumRegister(1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr, name="probe")
    qc.h(0)
    qc.measure(0, cr[0])

    payload = json.loads(
        qiskit_to_ionq(qc, tempo_qis_backend, passed_args={"shots": 100})
    )
    header = decompress_metadata_string(payload["metadata"]["qiskit_header"])
    assert header["name"] == "probe"
    assert header["n_qubits"] == 1
    assert header["memory_slots"] == 1


# ---------------------------------------------------------------------------
# Settings routing (verbatim, include_leakage, symmetry_verification)
# ---------------------------------------------------------------------------


def test_v2_verbatim_native_ok(tempo_backend):
    """A circuit containing only native gates + measure may submit with verbatim=True."""
    qc = QuantumCircuit(2, 2)
    qc.append(GPI2Gate(0.25), [0])
    qc.append(ZZGate(0.25), [0, 1])
    qc.measure([0, 1], [0, 1])

    payload = json.loads(
        qiskit_to_ionq(qc, tempo_backend, passed_args={"shots": 100, "verbatim": True})
    )
    assert payload["settings"]["verbatim"] is True


def test_v2_verbatim_qis_rejects(tempo_backend):
    """verbatim=True with a QIS gate (e.g. h) should raise IonQGateError client-side."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    with pytest.raises(IonQGateError) as exc:
        qiskit_to_ionq(qc, tempo_backend, passed_args={"shots": 100, "verbatim": True})
    assert exc.value.gate_name == "h"


def test_v2_include_leakage_flag(tempo_qis_backend):
    """include_leakage=True ends up at settings.include_leakage."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    payload = json.loads(
        qiskit_to_ionq(
            qc, tempo_qis_backend, passed_args={"shots": 50, "include_leakage": True}
        )
    )
    assert payload["settings"]["include_leakage"] is True


def test_v2_symmetry_under_em(tempo_qis_backend):
    """symmetry_verification rides inside settings.error_mitigation."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    payload = json.loads(
        qiskit_to_ionq(
            qc,
            tempo_qis_backend,
            passed_args={"shots": 50, "symmetry_verification": True},
        )
    )
    em_block = payload["settings"]["error_mitigation"]
    assert em_block["symmetry_verification"] is True


def test_v2_no_settings_default(tempo_qis_backend):
    """A plain submission should not emit a ``settings`` block at all."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    payload = json.loads(
        qiskit_to_ionq(qc, tempo_qis_backend, passed_args={"shots": 50})
    )
    assert "settings" not in payload


# ---------------------------------------------------------------------------
# Constraints / negative cases
# ---------------------------------------------------------------------------


def test_v2_rejects_multi_circuit(tempo_qis_backend):
    """The ``ionq.multi-circuit.v2`` schema is not yet defined; we error early."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    with pytest.raises(IonQJobError, match="Multi-circuit"):
        qiskit_to_ionq([qc, qc], tempo_qis_backend, passed_args={"shots": 50})


def test_v1_path_unchanged_for_aria(provider):
    """Sanity guard: Aria submissions still emit the v1 dict, not a QASM string."""
    backend = provider.get_backend("ionq_qpu.aria-1")
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    payload = json.loads(qiskit_to_ionq(qc, backend, passed_args={"shots": 50}))
    assert payload["type"] == "ionq.circuit.v1"
    assert isinstance(payload["input"], dict)
    assert payload["input"]["gateset"] == "qis"
