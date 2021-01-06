# -*- coding: utf-8 -*-
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

"""Test the qobj_to_ionq function."""

import json

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit_ionq_provider.helpers import (
    qiskit_to_ionq,
    compress_dict_to_metadata_string,
    decompress_metadata_string_to_dict,
)


def test_output_map__with_multiple_measurements_to_different_clbits(simulator_backend):
    """Test a full circuit

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    qc = QuantumCircuit(2, 2, name="test_name")
    qc.measure(0, 0)
    qc.measure(0, 1)
    ionq_json = qiskit_to_ionq(
        qc,
        simulator_backend.name(),
        passed_args={"shots": 200},
    )
    actual = json.loads(ionq_json)
    actual_metadata = actual.pop("metadata") or {}
    actual_output_map = json.loads(actual_metadata.pop("output_map") or "{}")

    assert actual_output_map == [0, 0]


def test_output_map__with_multiple_measurements_to_same_clbit(simulator_backend):
    """Test a full circuit

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    qc = QuantumCircuit(2, 2, name="test_name")
    qc.measure(0, 0)
    qc.measure(1, 0)
    ionq_json = qiskit_to_ionq(
        qc,
        simulator_backend.name(),
        passed_args={"shots": 200},
    )
    actual = json.loads(ionq_json)
    actual_metadata = actual.pop("metadata") or {}
    actual_output_map = json.loads(actual_metadata.pop("output_map") or "{}")

    assert actual_output_map == [1, None]


def test_metadata_header__with_multiple_registers(simulator_backend):
    """Test correct metadata headers when we have multiple qregs and cregs"""
    qr0 = QuantumRegister(2, "qr0")
    qr1 = QuantumRegister(2, "qr1")
    cr0 = ClassicalRegister(2, "cr0")
    cr1 = ClassicalRegister(2, "cr1")

    qc = QuantumCircuit(qr0, qr1, cr0, cr1)
    qc.measure([qr1[0], qr1[1]], [cr1[0], cr1[1]])

    expected_metadata_header = {
        "memory_slots": 2,
        "global_phase": 0,
        "n_qubits": 2,
        "name": "test_name",
        "creg_sizes": [["cr0", 2], ["cr1", 2]],
        "clbit_labels": [["cr0", 0], ["cr0", 1], ["cr1", 3], ["cr1", 4]],
        "qreg_sizes": [["qr0", 2]],
        "qubit_labels": [["qr0", 0], ["qr0", 1], ["qr1", 3], ["qr1", 4]],
    }


def test_full_circuit(simulator_backend):
    """Test a full circuit

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    qc = QuantumCircuit(2, 2, name="test_name")
    qc.cnot(1, 0)
    qc.h(1)
    qc.measure(1, 0)
    qc.measure(0, 1)
    ionq_json = qiskit_to_ionq(
        qc,
        simulator_backend.name(),
        passed_args={"shots": 200},
    )
    expected_metadata_header = {
        "memory_slots": 2,
        "global_phase": 0,
        "n_qubits": 2,
        "name": "test_name",
        "creg_sizes": [["c", 2]],
        "clbit_labels": [["c", 0], ["c", 1]],
        "qreg_sizes": [["q", 2]],
        "qubit_labels": [["q", 0], ["q", 1]],
    }
    expected_output_map = [1, 0]
    expected_metadata = {
        "shots": "200",
    }
    expected = {
        "lang": "json",
        "target": "simulator",
        "shots": 200,
        "body": {
            "qubits": 2,
            "circuit": [
                {"gate": "x", "controls": [1], "targets": [0]},
                {"gate": "h", "targets": [1]},
            ],
        },
    }

    actual = json.loads(ionq_json)
    actual_metadata = actual.pop("metadata") or {}
    actual_output_map = json.loads(actual_metadata.pop("output_map") or "{}")
    actual_metadata_header = decompress_metadata_string_to_dict(
        actual_metadata.pop("qiskit_header") or None
    )

    # check dict equality:
    assert actual == expected
    assert actual_metadata == expected_metadata
    assert actual_metadata_header == expected_metadata_header
    assert actual_output_map == expected_output_map
