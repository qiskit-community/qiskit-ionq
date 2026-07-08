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

# Copyright 2026 IonQ, Inc. (www.ionq.com)
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

"""Tests for the ionq_native output stage and native-gate equivalences."""

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import UGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator, random_unitary

from qiskit_ionq.ionq_gates import GPIGate, GPI2Gate, MSGate
from qiskit_ionq.ionq_native_stage import _native_1q_sequence


def test_parameterized_transpile(provider):
    """Unbound parameters transpile symbolically onto native gates."""
    backend = provider.get_backend("ionq_simulator", gateset="native")
    backend.set_options(noise_model="forte-1")
    theta = Parameter("θ")
    qc = QuantumCircuit(2)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    transpiled = transpile(
        qc,
        backend=backend,
        optimization_level=1,
        seed_transpiler=0,
        initial_layout=[0, 1],
    )
    assert set(transpiled.count_ops()) <= {"gpi", "gpi2", "zz"}
    dag = circuit_to_dag(transpiled)  # drop idle padding qubits
    dag.remove_qubits(*(w for w in dag.idle_wires() if w in dag.qubits))
    transpiled = dag_to_circuit(dag)
    for value in (0.37, -1.2):
        bound = transpiled.assign_parameters({next(iter(transpiled.parameters)): value})
        assert Operator(bound).equiv(Operator(qc.assign_parameters({theta: value})))


def test_native_1q_sequence():
    """Numeric 1q synthesis is exact (global phase included) and <= 3 gates."""
    samples = [np.eye(2, dtype=complex), np.diag([np.exp(0.7j), np.exp(-0.1j)])]
    samples += [np.asarray(GPIGate(0.3)), np.asarray(GPI2Gate(-0.2))]
    samples += [random_unitary(2, seed=seed).data for seed in range(25)]
    for matrix in samples:
        gates, phase = _native_1q_sequence(matrix)
        assert len(gates) <= 3
        composed = np.eye(2, dtype=complex)
        for gate in gates:
            composed = np.asarray(gate.to_matrix()) @ composed
        np.testing.assert_allclose(np.exp(1j * phase) * composed, matrix, atol=1e-9)


def test_equivalence_rules_exact():
    """The registered U->GPI2.GPI.GPI2 and MS->RZ.RXX.RZ rules are exact,
    global phase included (required by the equivalence library)."""
    for gate, marker in [
        (UGate(0.7, 1.9, -0.4), "gpi"),
        (MSGate(0.1, 0.2, 0.22), "rxx"),
        (MSGate(0, 0, 0.25), "rxx"),
    ]:
        entries = [
            circ
            for circ in SessionEquivalenceLibrary.get_entry(gate)
            if any(inst.operation.name == marker for inst in circ.data)
        ]
        assert entries, f"IonQ equivalence for {gate.name} not registered"
        for circ in entries:
            np.testing.assert_allclose(
                Operator(circ).data, Operator(gate).data, atol=1e-9
            )
