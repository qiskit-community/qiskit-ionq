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

"""Test the qiskit_to_ionq function."""

import json
import pytest

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_ionq.exceptions import IonQGateError
from qiskit_ionq.helpers import qiskit_to_ionq, decompress_metadata_string_to_dict
from qiskit_ionq.ionq_gates import GPIGate, GPI2Gate, MSGate
from qiskit_ionq.constants import ErrorMitigation


def test_output_map__with_multiple_measurements_to_different_clbits(
    simulator_backend,
):  # pylint: disable=invalid-name
    """
    Test output mapping handles multiple measurements from the same qubit to
    different clbits correctly

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    qc = QuantumCircuit(2, 2, name="test_name")
    qc.measure(0, 0)
    qc.measure(0, 1)
    ionq_json = qiskit_to_ionq(
        qc, simulator_backend, passed_args={"shots": 200, "sampler_seed": 42}
    )
    actual = json.loads(ionq_json)
    actual_maps = actual.pop("registers") or {}
    actual_output_map = actual_maps.pop("meas_mapped")

    assert actual_output_map == [0, 0]


def test_output_map__with_multiple_measurements_to_same_clbit(
    simulator_backend,
):  # pylint: disable=invalid-name
    """Test output mapping handles multiple measurements to same clbit correctly

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    qc = QuantumCircuit(2, 2, name="test_name")
    qc.measure(0, 0)
    qc.measure(1, 0)
    ionq_json = qiskit_to_ionq(
        qc, simulator_backend, passed_args={"shots": 200, "sampler_seed": 42}
    )
    actual = json.loads(ionq_json)
    actual_maps = actual.pop("registers") or {}
    actual_output_map = actual_maps.pop("meas_mapped")

    assert actual_output_map == [1, None]


def test_output_map__with_multiple_registers(
    simulator_backend,
):  # pylint: disable=invalid-name
    """Test output map with multiple registers

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    qr0 = QuantumRegister(2, "qr0")
    qr1 = QuantumRegister(2, "qr1")
    cr0 = ClassicalRegister(2, "cr0")
    cr1 = ClassicalRegister(2, "cr1")

    qc = QuantumCircuit(qr0, qr1, cr0, cr1, name="test_name")
    qc.measure([qr0[0], qr0[1], qr1[0], qr1[1]],
               [cr0[0], cr0[1], cr1[0], cr1[1]])

    ionq_json = qiskit_to_ionq(
        qc, simulator_backend, passed_args={"shots": 123, "sampler_seed": 42}
    )
    actual = json.loads(ionq_json)
    actual_maps = actual.pop("registers") or {}
    actual_output_map = actual_maps.pop("meas_mapped")

    assert actual_output_map == [0, 1, 2, 3]


def test_metadata_header__with_multiple_registers(
    simulator_backend,
):  # pylint: disable=invalid-name
    """Test correct metadata headers when we have multiple qregs and cregs"""
    qr0 = QuantumRegister(2, "qr0")
    qr1 = QuantumRegister(2, "qr1")
    cr0 = ClassicalRegister(2, "cr0")
    cr1 = ClassicalRegister(2, "cr1")

    qc = QuantumCircuit(qr0, qr1, cr0, cr1, name="test_name")
    qc.measure([qr1[0], qr1[1]], [cr1[0], cr1[1]])

    ionq_json = qiskit_to_ionq(
        qc, simulator_backend, passed_args={"shots": 200, "sampler_seed": 42}
    )

    expected_metadata_header = {
        "memory_slots": 4,
        "global_phase": 0,
        "n_qubits": 4,
        "name": "test_name",
        "creg_sizes": [["cr0", 2], ["cr1", 2]],
        "clbit_labels": [["cr0", 0], ["cr0", 1], ["cr1", 0], ["cr1", 1]],
        "qreg_sizes": [["qr0", 2], ["qr1", 2]],
        "qubit_labels": [["qr0", 0], ["qr0", 1], ["qr1", 0], ["qr1", 1]],
    }

    actual = json.loads(ionq_json)
    actual_metadata = actual.pop("metadata") or {}
    actual_metadata_header = decompress_metadata_string_to_dict(
        actual_metadata.pop("qiskit_header") or None
    )

    # check dict equality:
    assert actual_metadata_header == expected_metadata_header


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
        qc, simulator_backend, passed_args={"shots": 200, "sampler_seed": 42}
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
    expected_metadata = {"shots": "200", "sampler_seed": "42"}
    expected_rest_of_payload = {
        "target": "simulator",
        "shots": 200,
        "name": "test_name",
        "noise": {
            "model": "ideal",
            "seed": None,
        },
        "input": {
            "format": "ionq.circuit.v0",
            "gateset": "qis",
            "qubits": 2,
            "circuit": [
                {"gate": "x", "controls": [1], "targets": [0]},
                {"gate": "h", "targets": [1]},
            ],
        },
    }

    actual = json.loads(ionq_json)
    actual_metadata = actual.pop("metadata") or {}
    actual_metadata_header = decompress_metadata_string_to_dict(
        actual_metadata.pop("qiskit_header") or None
    )
    actual_maps = actual.pop("registers") or {}
    actual_output_map = actual_maps.pop("meas_mapped") or []

    # check dict equality:
    assert actual_metadata == expected_metadata
    assert actual_metadata_header == expected_metadata_header
    assert actual_output_map == expected_output_map
    assert actual == expected_rest_of_payload


def test_circuit_transpile(simulator_backend):
    """Test a full circuit on a native backend via transpilation

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    new_backend = simulator_backend.with_name(
        "ionq_simulator", gateset="native")
    circ = QuantumCircuit(2, 2, name="blame_test")
    circ.cnot(1, 0)
    circ.h(1)
    circ.measure(1, 0)
    circ.measure(0, 1)

    with pytest.raises(TranspilerError) as exc_info:
        transpile(circ, backend=new_backend)
    assert "Unable to map source basis" in exc_info.value.message


def test_circuit_incorrect(simulator_backend):
    """Test a full circuit on a native backend

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    native_backend = simulator_backend.with_name(
        "ionq_simulator", gateset="native")
    circ = QuantumCircuit(2, 2, name="blame_test")
    circ.cnot(1, 0)
    circ.h(1)
    circ.measure(1, 0)
    circ.measure(0, 1)
    with pytest.raises(IonQGateError) as exc_info:
        qiskit_to_ionq(
            circ,
            native_backend,
            passed_args={"shots": 200, "sampler_seed": 23},
        )
    assert exc_info.value.gateset == "native"


def test_native_circuit_incorrect(simulator_backend):
    """Test a full native circuit on a QIS backend

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    circ = QuantumCircuit(3, name="blame_test")
    circ.append(GPIGate(0.1), [0])
    circ.append(GPI2Gate(0.2), [1])
    with pytest.raises(IonQGateError) as exc_info:
        qiskit_to_ionq(
            circ,
            simulator_backend,
            passed_args={"shots": 200, "sampler_seed": 23},
        )
    assert exc_info.value.gateset == "qis"
    assert exc_info.value.gate_name == "gpi"


def test_native_circuit_transpile(simulator_backend):
    """Test a full native circuit on a QIS backend via transpilation

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    circ = QuantumCircuit(3, name="blame_test")
    circ.append(GPIGate(0.1), [0])
    circ.append(GPI2Gate(0.2), [1])
    circ.append(MSGate(0.2, 0.3, 0.25), [1, 2])

    with pytest.raises(QiskitError) as exc_info:
        transpile(circ, backend=simulator_backend)
    assert "Cannot unroll the circuit to the given basis" in exc_info.value.message


def test_full_native_circuit(simulator_backend):
    """Test a full native circuit

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
    """
    native_backend = simulator_backend.with_name(
        "ionq_simulator", gateset="native")
    qc = QuantumCircuit(3, name="blame_test")
    qc.append(GPIGate(0.1), [0])
    qc.append(GPI2Gate(0.2), [1])
    qc.append(MSGate(0.2, 0.3, 0.25), [1, 2])
    ionq_json = qiskit_to_ionq(
        qc,
        native_backend,
        passed_args={
            "noise_model": "harmony",
            "sampler_seed": 23,
            "shots": 200
        },
    )
    expected_metadata_header = {
        "memory_slots": 0,
        "global_phase": 0,
        "n_qubits": 3,
        "name": "blame_test",
        "creg_sizes": [],
        "clbit_labels": [],
        "qreg_sizes": [["q", 3]],
        "qubit_labels": [["q", 0], ["q", 1], ["q", 2]],
    }
    expected_metadata = {"shots": "200", "sampler_seed": "23"}
    expected_rest_of_payload = {
        "target": "simulator",
        "name": "blame_test",
        "shots": 200,
        "noise": {
            "model": "harmony",
            "seed": None,
        },
        "input": {
            "format": "ionq.circuit.v0",
            "gateset": "native",
            "qubits": 3,
            "circuit": [
                {"gate": "gpi", "target": 0, "phase": 0.1},
                {"gate": "gpi2", "target": 1, "phase": 0.2},
                {"gate": "ms", "targets": [1, 2],
                    "phases": [0.2, 0.3], "angle": 0.25},
            ],
        },
    }

    actual = json.loads(ionq_json)
    actual_metadata = actual.pop("metadata") or {}
    actual_metadata_header = decompress_metadata_string_to_dict(
        actual_metadata.pop("qiskit_header") or None
    )
    registers = actual.pop("registers") or {}

    # check dict equality:
    assert actual_metadata == expected_metadata
    assert actual_metadata_header == expected_metadata_header
    assert "meas_mapped" not in registers
    assert actual == expected_rest_of_payload


@pytest.mark.parametrize(
    "error_mitigation,expected",
    [
        (ErrorMitigation.NO_SYMMETRIZATION, {"symmetrization": False}),
        (ErrorMitigation.SYMMETRIZATION, {"symmetrization": True}),
    ],
)
def test__error_mitigation_settings(simulator_backend, error_mitigation, expected):
    """Test error_mitigation settings get serialized accordingly

    Args:
        simulator_backend (IonQSimulatorBackend): A simulator backend fixture.
        error_mitigation (ErrorMitigation): error mitigation setting
        expected (dict): expected serialization
    """
    qc = QuantumCircuit(1, 1)

    args = {
        "shots": 123,
        "sampler_seed": 42,
        "error_mitigation": error_mitigation
    }
    ionq_json = qiskit_to_ionq(
        qc, simulator_backend, passed_args=args
    )
    actual = json.loads(ionq_json)
    actual_error_mitigation = actual.pop("error_mitigation")

    assert actual_error_mitigation == expected
