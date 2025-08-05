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

import math
from sympy import Matrix, exp, sqrt

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler.basepasses import TransformationPass

from .ionq_gates import GPIGate, GPI2Gate


class CancelGPI2Adjoint(TransformationPass):
    """GPI2 times GPI2 adjoint should cancel."""

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
                        if math.isclose((phi2 + 0.5) % 1, phi1 % 1) or math.isclose(
                            (phi1 + 0.5) % 1, phi2 % 1
                        ):
                            nodes_to_remove.extend([node, next_node])

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class CancelGPIAdjoint(TransformationPass):
    """GPI times GPI should cancel."""

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
                        if math.isclose(phi1, phi2):
                            nodes_to_remove.extend([node, next_node])

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class GPI2TwiceIsGPI(TransformationPass):
    """GPI2 times GPI2 is GPI times -i. Below the -i factor will be ignored."""

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
                        if math.isclose(phi1, phi2):
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
    """More than three single qubit gates in series are collapsed to 3 gates."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []
        visited_nodes = []

        for node in dag.topological_op_nodes():
            if (node.op.name in ("gpi", "gpi2")) and node not in visited_nodes:
                single_qubit_gates_streak = self.get_streak_recursively(dag, [node])
                if len(single_qubit_gates_streak) > 3:
                    self.compact_single_qubits_streak(dag, single_qubit_gates_streak)
                    nodes_to_remove.extend(single_qubit_gates_streak[:-1])
                    visited_nodes.extend(single_qubit_gates_streak[:-1])
                else:
                    visited_nodes.extend(single_qubit_gates_streak)

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag

    def get_streak_recursively(self, dag, streak):
        """Recursively build up single qubit streak."""
        last_node = streak[-1]
        successors = [
            succ
            for succ in dag.quantum_successors(last_node)
            if isinstance(succ, DAGOpNode)
        ]
        for node in successors:
            if (node.op.name in ("gpi", "gpi2")) and last_node.qargs == node.qargs:
                streak.append(node)
                return self.get_streak_recursively(dag, streak)
        return streak

    def multiply_node_matrices(self, nodes: list) -> Matrix:
        """Compute matrix multiplication."""
        matrix = Matrix([[1, 0], [0, 1]])
        for node in nodes:
            if node.op.name == "gpi":
                phi = node.op.params[0]
                matrix = (
                    Matrix(
                        [
                            [0, exp(-1j * phi * 2 * math.pi)],
                            [exp(1j * phi * 2 * math.pi), 0],
                        ]
                    )
                    * matrix
                )
            if node.op.name == "gpi2":
                phi = node.op.params[0]
                matrix = (
                    Matrix(
                        [
                            [1, -1j * exp(-1j * phi * 2 * math.pi)],
                            [-1j * exp(1j * phi * 2 * math.pi), 1],
                        ]
                    )
                    * matrix
                    * (1 / sqrt(2))
                )
        return matrix

    def get_euler_angles(self, matrix: Matrix) -> tuple:
        """Get Euler angles decomposition for an arbitrary matrix."""
        operator = Operator(matrix.tolist())
        decomposer = OneQubitEulerDecomposer("U3")
        theta, phi, lambd = decomposer.angles(operator)
        return (theta, phi, lambd)

    def compact_single_qubits_streak(self, dag, single_qubit_gates_streak):
        """Merge GPIs gates in a single qubit streak to 3 GPIs gates."""
        matrix = self.multiply_node_matrices(single_qubit_gates_streak)
        theta, phi, lambd = self.get_euler_angles(matrix)
        last_gate = single_qubit_gates_streak[-1]

        qc = QuantumCircuit(dag.num_qubits())
        qubit_index = dag.qubits.index(last_gate.qargs[0])
        qc.append(GPI2Gate(0.5 - lambd / (2 * math.pi)), [qubit_index])
        qc.append(
            GPIGate(
                theta / (4 * math.pi) + phi / (4 * math.pi) - lambd / (4 * math.pi)
            ),
            [qubit_index],
        )
        qc.append(GPI2Gate(0.5 + phi / (2 * math.pi)), [qubit_index])
        qc_dag = circuit_to_dag(qc)

        wire_mapping = {qarg: qarg for qarg in last_gate.qargs}
        dag.substitute_node_with_dag(last_gate, qc_dag, wires=wire_mapping)


class CommuteGPIsThroughMS(TransformationPass):
    """GPI(0), GPI(π), GPI(-π), GPI2(0), GPI2(π), GPI2(-π)
    on either qubit commute with MS"""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = set()

        for node in dag.topological_op_nodes():
            if node in nodes_to_remove:
                continue

            if (node.op.name in ("gpi", "gpi2")) and (
                math.isclose(node.op.params[0], 0)
                or math.isclose(node.op.params[0], 0.5)
                or math.isclose(node.op.params[0], -0.5)
            ):
                successors = [
                    succ for succ in dag.successors(node) if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if (
                        next_node.op.name == "ms"
                        and math.isclose(next_node.op.params[0], 0)
                        and math.isclose(next_node.op.params[1], 0)
                        and math.isclose(next_node.op.params[2], 0.25)
                        and node.qargs[0] in next_node.qargs
                    ):
                        sub_dag = DAGCircuit()
                        for qreg in dag.qregs.values():
                            sub_dag.add_qreg(qreg)

                        # map the ops to the qubits in the sub-DAG
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
