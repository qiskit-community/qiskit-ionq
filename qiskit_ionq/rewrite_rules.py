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

"""Rewrite rules for optimizing the transpilation to IonQ native gates."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler.basepasses import TransformationPass

from .ionq_gates import GPIGate, GPI2Gate, MSGate, ZZGate


class CancelGPI2Adjoint(TransformationPass):
    """GPI2 times GPI2 adjoint cancels."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []

        for node in dag.topological_op_nodes():
            if (
                isinstance(node, DAGOpNode)
                and node.op.name == "gpi2"
                and node not in nodes_to_remove
            ):
                successors = [
                    succ
                    for succ in dag.quantum_successors(node)
                    if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if next_node.op.name == "gpi2" and node.qargs == next_node.qargs:
                        phi1 = node.op.params[0]
                        phi2 = next_node.op.params[0]
                        if np.isclose((phi2 + 0.5) % 1, phi1 % 1) or np.isclose(
                            (phi1 + 0.5) % 1, phi2 % 1
                        ):
                            nodes_to_remove.extend([node, next_node])

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class CancelGPIAdjoint(TransformationPass):
    """GPI times GPI cancels."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []

        for node in dag.topological_op_nodes():
            if node.op.name == "gpi" and node not in nodes_to_remove:
                successors = [
                    succ
                    for succ in dag.quantum_successors(node)
                    if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if next_node.op.name == "gpi" and node.qargs == next_node.qargs:
                        phi1 = node.op.params[0]
                        phi2 = next_node.op.params[0]
                        if np.isclose(phi1, phi2):
                            nodes_to_remove.extend([node, next_node])

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class GPI2TwiceIsGPI(TransformationPass):
    """Two GPI2 operations compose to a GPI up to a global phase (ignored)."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []

        for node in dag.topological_op_nodes():
            if node.op.name == "gpi2" and node not in nodes_to_remove:
                successors = [
                    succ
                    for succ in dag.quantum_successors(node)
                    if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if next_node.op.name == "gpi2" and node.qargs == next_node.qargs:
                        phi1 = node.op.params[0]
                        phi2 = next_node.op.params[0]
                        if np.isclose(phi1, phi2):
                            qc = QuantumCircuit(dag.num_qubits())
                            qubit_index = dag.qubits.index(node.qargs[0])
                            qc.append(GPIGate(phi1), [qubit_index])
                            qc_dag = circuit_to_dag(qc)

                            wire_mapping = {qarg: qarg for qarg in next_node.qargs}

                            dag.substitute_node_with_dag(
                                next_node, qc_dag, wires=wire_mapping
                            )
                            nodes_to_remove.append(node)

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class CompactMoreThanThreeSingleQubitGates(TransformationPass):
    """Collapse series of >3 single-qubit gates into a 3-gate canonical form."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []
        visited_nodes = []

        for node in dag.topological_op_nodes():
            if (node.op.name in ("gpi", "gpi2")) and node not in visited_nodes:
                single_qubit_gates_streak = self._get_streak_recursively(dag, [node])
                if len(single_qubit_gates_streak) > 3:
                    self._compact_single_qubits_streak(dag, single_qubit_gates_streak)
                    nodes_to_remove.extend(single_qubit_gates_streak[:-1])
                    visited_nodes.extend(single_qubit_gates_streak[:-1])
                else:
                    visited_nodes.extend(single_qubit_gates_streak)

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag

    def _get_streak_recursively(self, dag: DAGCircuit, streak: list) -> list:
        """Recursively build up a streak of single-qubit GPI/GPI2 gates."""
        last_node = streak[-1]
        successors = [
            succ
            for succ in dag.quantum_successors(last_node)
            if isinstance(succ, DAGOpNode)
        ]
        for node in successors:
            if (node.op.name in ("gpi", "gpi2")) and last_node.qargs == node.qargs:
                streak.append(node)
                return self._get_streak_recursively(dag, streak)
        return streak

    def _multiply_node_matrices(self, nodes: list) -> np.ndarray:
        """Return the product of all gate matrices in nodes as a 2x2 NumPy array."""
        matrix = np.eye(2, dtype=complex)
        for node in nodes:
            phi = node.op.params[0]
            if node.op.name == "gpi":
                gate = np.array(
                    [
                        [0, np.exp(-1j * phi * 2 * np.pi)],
                        [np.exp(1j * phi * 2 * np.pi), 0],
                    ],
                    dtype=complex,
                )
            elif node.op.name == "gpi2":
                gate = np.array(
                    [
                        [1, -1j * np.exp(-1j * phi * 2 * np.pi)],
                        [-1j * np.exp(1j * phi * 2 * np.pi), 1],
                    ],
                    dtype=complex,
                ) / np.sqrt(2)
            else:
                continue
            matrix = gate @ matrix  # Left-multiplication accumulates the product.
        return matrix

    def _get_euler_angles(self, matrix: np.ndarray) -> tuple[float, float, float]:
        """Return (θ, φ, λ) Euler angles for the given 2x2 unitary matrix."""
        operator = Operator(matrix)
        decomposer = OneQubitEulerDecomposer("U3")
        theta, phi, lambd = decomposer.angles(operator)
        return theta, phi, lambd

    def _compact_single_qubits_streak(
        self, dag: DAGCircuit, single_qubit_gates_streak: list
    ) -> None:
        """Merge a streak of GPI/GPI2 gates into three gates in-place on dag."""
        matrix = self._multiply_node_matrices(single_qubit_gates_streak)
        theta, phi, lambd = self._get_euler_angles(matrix)
        last_gate = single_qubit_gates_streak[-1]

        qc = QuantumCircuit(dag.num_qubits())
        qubit_index = dag.qubits.index(last_gate.qargs[0])
        qc.append(GPI2Gate(0.5 - lambd / (2 * np.pi)), [qubit_index])
        qc.append(
            GPIGate(theta / (4 * np.pi) + phi / (4 * np.pi) - lambd / (4 * np.pi)),
            [qubit_index],
        )
        qc.append(GPI2Gate(0.5 + phi / (2 * np.pi)), [qubit_index])
        qc_dag = circuit_to_dag(qc)

        wire_mapping = {qarg: qarg for qarg in last_gate.qargs}
        dag.substitute_node_with_dag(last_gate, qc_dag, wires=wire_mapping)


class CommuteGPIsThroughMS(TransformationPass):
    """Recognize that certain Clifford-phase GPIs commute with MS entangling gates."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = set()

        for node in dag.topological_op_nodes():
            if node in nodes_to_remove:
                continue

            if (node.op.name in ("gpi", "gpi2")) and (
                np.isclose(node.op.params[0], 0)
                or np.isclose(node.op.params[0], 0.5)
                or np.isclose(node.op.params[0], -0.5)
            ):
                successors = [
                    succ for succ in dag.successors(node) if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if (
                        next_node.op.name == "ms"
                        and np.isclose(next_node.op.params[0], 0)
                        and np.isclose(next_node.op.params[1], 0)
                        and np.isclose(next_node.op.params[2], 0.25)
                        and node.qargs[0] in next_node.qargs
                    ):
                        sub_dag = DAGCircuit()
                        for qreg in dag.qregs.values():
                            sub_dag.add_qreg(qreg)

                        # Map the ops to the qubits in the sub-DAG.
                        ms_qubits = [next_node.qargs[0], next_node.qargs[1]]
                        gpis_qubit = [node.qargs[0]]

                        sub_dag.apply_operation_back(next_node.op, ms_qubits)
                        sub_dag.apply_operation_back(node.op, gpis_qubit)

                        wire_mapping = {qubit: qubit for qubit in ms_qubits}
                        wire_mapping[node.qargs[0]] = node.qargs[0]

                        dag.substitute_node_with_dag(
                            next_node, sub_dag, wires=wire_mapping
                        )
                        nodes_to_remove.add(node)
                        break

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class CommuteGPIsThroughZZ(TransformationPass):
    """Recognize that certain Clifford-phase GPIs commute with ZZ entangling gates."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = set()

        for node in dag.topological_op_nodes():
            if node in nodes_to_remove:
                continue

            if node.op.name in ("gpi"):
                successors = [
                    succ for succ in dag.successors(node) if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if next_node.op.name != "zz":
                        continue

                    # Only meaningful if the GPI acts on one of the ZZ qubits.
                    if node.qargs[0] not in next_node.qargs:
                        continue

                    sub_dag = DAGCircuit()
                    for qreg in dag.qregs.values():
                        sub_dag.add_qreg(qreg)

                    zz_qubits = [next_node.qargs[0], next_node.qargs[1]]
                    gpi_qubit = [node.qargs[0]]

                    # Conjugation by X/Y (any GPI) flips Z on that qubit, so ZZ(θ) -> ZZ(-θ).
                    theta = next_node.op.params[0]
                    sub_dag.apply_operation_back(ZZGate(-theta), zz_qubits)
                    sub_dag.apply_operation_back(node.op, gpi_qubit)

                    wire_mapping = {qubit: qubit for qubit in zz_qubits}
                    wire_mapping[node.qargs[0]] = node.qargs[0]

                    dag.substitute_node_with_dag(next_node, sub_dag, wires=wire_mapping)
                    nodes_to_remove.add(node)
                    break

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class FuseConsecutiveMS(TransformationPass):
    """Fuse adjacent MS(φ0,φ1,θ1) then MS(φ0,φ1,θ2) on the same qubits into MS(φ0,φ1,θ1+θ2)."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        while changed:
            changed = False
            for node in list(dag.topological_op_nodes()):
                if node.op.name != "ms":
                    continue

                phi0, phi1, theta1 = node.op.params[:3]

                # First op-node successors touching each qubit of the pair.
                succs = []
                for q in node.qargs:
                    nxts = [
                        s
                        for s in dag.quantum_successors(node)
                        if isinstance(s, DAGOpNode) and q in s.qargs
                    ]
                    if nxts:
                        succs.append(nxts[0])

                if not (
                    len(succs) == 2
                    and succs[0] is succs[1]
                    and succs[0].op.name == "ms"
                    and set(succs[0].qargs) == set(node.qargs)
                ):
                    continue

                nxt = succs[0]
                phi0b, phi1b, theta2 = nxt.op.params[:3]
                if not (
                    np.isclose(phi0, phi0b, atol=1e-9)
                    and np.isclose(phi1, phi1b, atol=1e-9)
                ):
                    continue

                theta = theta1 + theta2
                theta = (theta + 0.5) % 1.0 - 0.5  # normalize to (-0.5, 0.5]

                if np.isclose(theta, 0.0, atol=1e-9):
                    dag.remove_op_node(nxt)
                    dag.remove_op_node(node)
                else:
                    sub = DAGCircuit()
                    for qreg in dag.qregs.values():
                        sub.add_qreg(qreg)
                    sub.apply_operation_back(MSGate(phi0, phi1, theta), list(nxt.qargs))
                    dag.substitute_node_with_dag(nxt, sub, {q: q for q in nxt.qargs})
                    dag.remove_op_node(node)

                changed = True
                break

        return dag


class FuseConsecutiveZZ(TransformationPass):
    """Fuse adjacent ZZ(θ1) then ZZ(θ2) on the same qubits into ZZ(θ1+θ2)."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        while changed:
            changed = False
            for node in list(dag.topological_op_nodes()):
                if node.op.name != "zz":
                    continue

                # First op-node successors touching each qubit of the pair.
                succs = []
                for q in node.qargs:
                    nxts = [
                        s
                        for s in dag.quantum_successors(node)
                        if isinstance(s, DAGOpNode) and q in s.qargs
                    ]
                    if nxts:
                        succs.append(nxts[0])

                if not (
                    len(succs) == 2
                    and succs[0] is succs[1]
                    and succs[0].op.name == "zz"
                    and set(succs[0].qargs) == set(node.qargs)
                ):
                    continue

                nxt = succs[0]
                theta = node.op.params[0] + nxt.op.params[0]
                theta = (theta + 0.5) % 1.0 - 0.5  # normalize to (-0.5, 0.5]

                if np.isclose(theta, 0.0, atol=1e-9):
                    dag.remove_op_node(nxt)
                    dag.remove_op_node(node)
                else:
                    sub = DAGCircuit()
                    for qreg in dag.qregs.values():
                        sub.add_qreg(qreg)
                    sub.apply_operation_back(ZZGate(theta), list(nxt.qargs))
                    dag.substitute_node_with_dag(nxt, sub, {q: q for q in nxt.qargs})
                    dag.remove_op_node(node)

                changed = True
                break

        return dag
