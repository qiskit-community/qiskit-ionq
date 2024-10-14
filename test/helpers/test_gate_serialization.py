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

import pytest
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    instruction,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from qiskit_ionq import exceptions
from qiskit_ionq.helpers import qiskit_circ_to_ionq_circ

compiler_directives = ["barrier"]
unsupported_instructions = [
    "reset",
    "initialize",
    "u",
    "custom-gate",
    "custom-gate-2",
]

gate_serializations = [
    ("ccx", [0, 1, 2], [{"gate": "x", "targets": [2], "controls": [0, 1]}]),
    ("ch", [0, 1], [{"gate": "h", "targets": [1], "controls": [0]}]),
    ("cx", [0, 1], [{"gate": "x", "targets": [1], "controls": [0]}]),
    (
        "cp",
        [0.5, 0, 1],
        [{"gate": "z", "rotation": 0.5, "targets": [1], "controls": [0]}],
    ),
    (
        "crx",
        [0, 0, 1],
        [{"gate": "rx", "rotation": 0, "targets": [1], "controls": [0]}],
    ),
    (
        "crx",
        [0.5, 0, 1],
        [{"gate": "rx", "rotation": 0.5, "targets": [1], "controls": [0]}],
    ),
    (
        "cry",
        [0, 0, 1],
        [{"gate": "ry", "rotation": 0, "targets": [1], "controls": [0]}],
    ),
    (
        "cry",
        [0.5, 0, 1],
        [{"gate": "ry", "rotation": 0.5, "targets": [1], "controls": [0]}],
    ),
    (
        "crz",
        [0, 0, 1],
        [{"gate": "rz", "rotation": 0, "targets": [1], "controls": [0]}],
    ),
    (
        "crz",
        [0.5, 0, 1],
        [{"gate": "rz", "rotation": 0.5, "targets": [1], "controls": [0]}],
    ),
    ("csx", [0, 1], [{"gate": "v", "targets": [1], "controls": [0]}]),
    ("cx", [0, 1], [{"gate": "x", "targets": [1], "controls": [0]}]),
    ("cy", [0, 1], [{"gate": "y", "targets": [1], "controls": [0]}]),
    ("cz", [0, 1], [{"gate": "z", "targets": [1], "controls": [0]}]),
    ("h", [0], [{"gate": "h", "targets": [0]}]),
    ("id", [0], []),
    (
        "mcp",
        [0.5, [0, 1], 2],
        [{"gate": "z", "rotation": 0.5, "targets": [2], "controls": [0, 1]}],
    ),
    ("mcx", [[0, 1], 2], [{"gate": "x", "targets": [2], "controls": [0, 1]}]),
    # make sure that multi-control can take any number of controls
    ("mcx", [[0, 1, 2], 3], [{"gate": "x", "targets": [3], "controls": [0, 1, 2]}]),
    (
        "mcx",
        [[0, 1, 2, 3], 4],
        [{"gate": "x", "targets": [4], "controls": [0, 1, 2, 3]}],
    ),
    (
        "mcx",
        [[0, 1, 2, 3, 4], 5],
        [{"gate": "x", "targets": [5], "controls": [0, 1, 2, 3, 4]}],
    ),
    ("measure", [0, 0], []),
    ("p", [0, 0], [{"gate": "z", "rotation": 0, "targets": [0]}]),
    ("p", [0.5, 0], [{"gate": "z", "rotation": 0.5, "targets": [0]}]),
    ("rx", [0, 0], [{"gate": "rx", "rotation": 0, "targets": [0]}]),
    ("rx", [0.5, 0], [{"gate": "rx", "rotation": 0.5, "targets": [0]}]),
    ("rxx", [0, 0, 1], [{"gate": "xx", "rotation": 0, "targets": [0, 1]}]),
    ("rxx", [0.5, 0, 1], [{"gate": "xx", "rotation": 0.5, "targets": [0, 1]}]),
    ("ry", [0, 0], [{"gate": "ry", "rotation": 0, "targets": [0]}]),
    ("ry", [0.5, 0], [{"gate": "ry", "rotation": 0.5, "targets": [0]}]),
    ("ryy", [0, 0, 1], [{"gate": "yy", "rotation": 0, "targets": [0, 1]}]),
    ("ryy", [0.5, 0, 1], [{"gate": "yy", "rotation": 0.5, "targets": [0, 1]}]),
    ("rz", [0, 0], [{"gate": "rz", "rotation": 0, "targets": [0]}]),
    ("rz", [0.5, 0], [{"gate": "rz", "rotation": 0.5, "targets": [0]}]),
    ("s", [0], [{"gate": "s", "targets": [0]}]),
    ("sdg", [0], [{"gate": "si", "targets": [0]}]),
    ("swap", [0, 1], [{"gate": "swap", "targets": [0, 1]}]),
    ("sx", [0], [{"gate": "v", "targets": [0]}]),
    ("sxdg", [0], [{"gate": "vi", "targets": [0]}]),
    ("t", [0], [{"gate": "t", "targets": [0]}]),
    ("tdg", [0], [{"gate": "ti", "targets": [0]}]),
    ("x", [0], [{"gate": "x", "targets": [0]}]),
    ("y", [0], [{"gate": "y", "targets": [0]}]),
    ("z", [0], [{"gate": "z", "targets": [0]}]),
]


@pytest.mark.parametrize("directive", compiler_directives)
def test_compiler_directives(directive):
    """Test that compiler directives are skipped.

    Args:
        directive (str): A compiler directive name.
    """
    unsupported = instruction.Instruction(directive, 0, 0, [])
    qc = QuantumCircuit(1, 1)
    qc.append(unsupported)
    circuit, _, _ = qiskit_circ_to_ionq_circ(qc)
    instruction_names = [instruction["gate"] for instruction in circuit]
    assert directive not in instruction_names


@pytest.mark.parametrize("instruction_name", unsupported_instructions)
def test_unsupported_instructions(instruction_name):
    """Test that trying to create a circuit that has an unsupported instruction
    results in an error.


    Args:
        instruction_name (str): an unsupported instruction name.
    """
    unsupported = instruction.Instruction(instruction_name, 0, 0, [])
    qc = QuantumCircuit(1, 1)
    qc.append(unsupported)
    with pytest.raises(exceptions.IonQGateError) as exc:
        qiskit_circ_to_ionq_circ(qc)
    assert exc.value.gate_name == unsupported.name


@pytest.mark.parametrize(
    "gate_name, gate_args, expected_serialization", gate_serializations
)
def test_individual_instruction_serialization(
    gate_name, gate_args, expected_serialization
):  # pylint: disable=invalid-name
    """Test that individual gates are correctly serialized

    Args:
        gate_name (str): the qiskit gate to test, as its QuantumCircuit method name
        gate_args (list): the arguments for the gate. Will have different
            entries and cardinality depending on the gate
        expected_serialization (list): expected serialization of the gate as a
            (normally single-element) list of instructions in IonQ API JSON format

    """
    qc = QuantumCircuit(6, 6)
    getattr(qc, gate_name)(*gate_args)
    serialized, _, _ = qiskit_circ_to_ionq_circ(qc)
    assert serialized == expected_serialization


def test_measurement_only_circuit():
    """Test a valid circuit with only measurements."""
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    expected = []
    built, _, _ = qiskit_circ_to_ionq_circ(qc)
    assert built == expected


def test_simple_circuit():
    """Test basic structure of a simple circuit"""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    expected = [{"gate": "h", "targets": [0]}]
    built, _, _ = qiskit_circ_to_ionq_circ(qc)
    assert built == expected


# pylint: disable=invalid-name
def test_circuit_with_entangling_ops():
    """Test structure of circuits with entangling ops."""
    qc = QuantumCircuit(2, 2)
    qc.cx(1, 0)
    expected = [{"gate": "x", "targets": [0], "controls": [1]}]
    built, _, _ = qiskit_circ_to_ionq_circ(qc)
    assert built == expected


def test_pauliexp_circuit():
    """Test structure of circuits with a Pauli evolution gate."""
    # build the evolution gate
    operator = SparsePauliOp(["XX", "YY", "ZZ"], coeffs=[0.1, 0.2, 0.3])
    evo = PauliEvolutionGate(operator, time=0.4)
    # append it to a circuit
    circuit = QuantumCircuit(3)
    circuit.append(evo, [1, 2])
    expected = [
        {
            "gate": "pauliexp",
            "targets": [1, 2],
            "terms": ["XX", "YY", "ZZ"],
            "coefficients": [0.1, 0.2, 0.3],
            "time": 0.4,
        }
    ]
    built, _, _ = qiskit_circ_to_ionq_circ(circuit)
    assert built == expected


@pytest.mark.parametrize("ionq_compiler_synthesis", [True, False])
def test_non_commuting_pauliexp_circuit(ionq_compiler_synthesis):
    """Test that non-commuting Pauli evolution gates raise an error."""
    # build the evolution gate
    operator = SparsePauliOp(["XX", "XY"], coeffs=[0.1, 0.2])
    evo = PauliEvolutionGate(operator, time=0.3)
    # append it to a circuit
    circuit = QuantumCircuit(2)
    circuit.append(evo, [0, 1])
    if ionq_compiler_synthesis:
        qiskit_circ_to_ionq_circ(
            circuit, ionq_compiler_synthesis=ionq_compiler_synthesis
        )
    else:
        with pytest.raises(exceptions.IonQPauliExponentialError) as _:
            qiskit_circ_to_ionq_circ(
                circuit, ionq_compiler_synthesis=ionq_compiler_synthesis
            )


def test_multi_control():
    """Test structure of circuits with multiple controls"""
    qc = QuantumCircuit(3, 3)
    qc.ccx(0, 1, 2)
    expected = [{"gate": "x", "targets": [2], "controls": [0, 1]}]
    built, _, _ = qiskit_circ_to_ionq_circ(qc)
    assert built == expected


def test_rotation_from_instruction_params():
    """Test that instruction parameters are used for rotation."""
    qc = QuantumCircuit(2, 2)
    qc.append(instruction.Instruction("rx", 2, 0, [1.0]), [1, 0])
    built, _, _ = qiskit_circ_to_ionq_circ(qc)
    built = built[0]
    assert "rotation" in built
    assert built["rotation"] == 1.0


def test_no_mid_circuit_measurement():
    """
    Test that putting an instruction on a qubit that has been measured is an
    invalid instruction
    """
    qc = QuantumCircuit(2, 2)
    qc.measure(1, 1)
    qc.x(1)
    with pytest.raises(exceptions.IonQMidCircuitMeasurementError) as exc:
        qiskit_circ_to_ionq_circ(qc)
    assert exc.value.qubit_index == 1
    assert exc.value.gate_name == "x"


def test_unordered_instructions_are_not_mid_circuit_measurement():
    """Test that mid-circuit measurement is only an error if
    you try and put an instruction on a measured qubit."""
    qc = QuantumCircuit(2, 2)
    qc.measure(1, 1)
    qc.x(0)
    expected = [{"gate": "x", "targets": [0]}]
    built, _, _ = qiskit_circ_to_ionq_circ(qc)
    assert built == expected


def test_circuit_with_multiple_registers():
    """Test that mid-circuit measurement is only an error if
    you try and put an instruction on a measured qubit."""
    qr0 = QuantumRegister(2, "qr0")
    qr1 = QuantumRegister(2, "qr1")
    cr0 = ClassicalRegister(2, "cr0")
    cr1 = ClassicalRegister(2, "cr1")

    qc = QuantumCircuit(
        qr0,
        qr1,
        cr0,
        cr1,
    )

    qc.x(qr0[0])
    qc.h(qr0[1])
    qc.y(qr1[0])
    qc.z(qr1[1])
    qc.cx(qr0[0], qr1[0])
    qc.measure([qr0[0], qr0[1], qr1[0], qr1[1]], [cr0[0], cr0[1], cr1[0], cr1[1]])

    expected = [
        {"gate": "x", "targets": [0]},
        {"gate": "h", "targets": [1]},
        {"gate": "y", "targets": [2]},
        {"gate": "z", "targets": [3]},
        {"gate": "x", "controls": [0], "targets": [2]},
    ]
    built, _, _ = qiskit_circ_to_ionq_circ(qc)
    assert built == expected
