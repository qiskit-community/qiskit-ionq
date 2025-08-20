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


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _mod1_turns(float_x: float) -> float:
    """Map a real to (-0.5, 0.5] in 'turns' (1.0 == 2π)."""
    return (float(float_x) + 0.5) % 1.0 - 0.5


def _near(float_a: float, float_b: float, tol: float = 1e-9) -> bool:
    return np.isclose(float_a, float_b, atol=tol)


def _same_qubits(node_1: DAGOpNode, node_2: DAGOpNode) -> bool:
    return set(node_1.qargs) == set(node_2.qargs)


def _first_succ_ops_on_qubits(
    dag: DAGCircuit, node: DAGOpNode, qubits
) -> list[DAGOpNode]:
    """Return the first op-node successors on each given qubit (if any)."""
    res = []
    for q in qubits:
        nxt = [
            s
            for s in dag.quantum_successors(node)
            if isinstance(s, DAGOpNode) and (q in s.qargs)
        ]
        if nxt:
            res.append(nxt[0])
    return res


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


class NormalizeNativeAngles(TransformationPass):
    """
    Canonicalize native-gate parameters:
      - GPI/GPI2 phases φ → (-0.5, 0.5]
      - ZZ and MS angles θ → (-0.5, 0.5] (drop if ~0)
      - MS phases (φ0, φ1) → (-0.5, 0.5]
    Only numeric parameters are normalized (symbolic left untouched).
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        to_remove = []
        for node in dag.topological_op_nodes():
            name = node.op.name
            if name == "gpi" and _is_number(node.op.params[0]):
                phi = _mod1_turns(node.op.params[0])
                if not _near(phi, node.op.params[0]):
                    sub = DAGCircuit()
                    for qreg in dag.qregs.values():
                        sub.add_qreg(qreg)
                    sub.apply_operation_back(GPIGate(phi), [node.qargs[0]])
                    dag.substitute_node_with_dag(
                        node, sub, {node.qargs[0]: node.qargs[0]}
                    )

            elif name == "gpi2" and _is_number(node.op.params[0]):
                phi = _mod1_turns(node.op.params[0])
                if not _near(phi, node.op.params[0]):
                    sub = DAGCircuit()
                    for qreg in dag.qregs.values():
                        sub.add_qreg(qreg)
                    sub.apply_operation_back(GPI2Gate(phi), [node.qargs[0]])
                    dag.substitute_node_with_dag(
                        node, sub, {node.qargs[0]: node.qargs[0]}
                    )

            elif name == "zz" and _is_number(node.op.params[0]):
                theta = _mod1_turns(node.op.params[0])
                if _near(theta, 0.0):
                    to_remove.append(node)
                elif not _near(theta, node.op.params[0]):
                    sub = DAGCircuit()
                    for qreg in dag.qregs.values():
                        sub.add_qreg(qreg)
                    sub.apply_operation_back(ZZGate(theta), list(node.qargs))
                    dag.substitute_node_with_dag(node, sub, {q: q for q in node.qargs})

            elif name == "ms" and all(_is_number(p) for p in node.op.params[:3]):
                phi0, phi1, theta = node.op.params[:3]
                phi0_n = _mod1_turns(phi0)
                phi1_n = _mod1_turns(phi1)
                theta_n = _mod1_turns(theta)
                if _near(theta_n, 0.0):
                    to_remove.append(node)
                elif (
                    (not _near(phi0, phi0_n))
                    or (not _near(phi1, phi1_n))
                    or (not _near(theta, theta_n))
                ):
                    sub = DAGCircuit()
                    for qreg in dag.qregs.values():
                        sub.add_qreg(qreg)
                    sub.apply_operation_back(
                        MSGate(phi0_n, phi1_n, theta_n), list(node.qargs)
                    )
                    dag.substitute_node_with_dag(node, sub, {q: q for q in node.qargs})

        for n in to_remove:
            dag.remove_op_node(n)
        return dag


class FuseConsecutiveZZ(TransformationPass):
    """Fuse adjacent ZZ(θ1) then ZZ(θ2) on the same qubits into ZZ(θ1+θ2)."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        while changed:
            changed = False
            for node in list(dag.topological_op_nodes()):
                if node.op.name != "zz" or not _is_number(node.op.params[0]):
                    continue
                qargs = node.qargs
                succs = _first_succ_ops_on_qubits(dag, node, qargs)
                if (
                    len(succs) == 2
                    and succs[0] is succs[1]
                    and succs[0].op.name == "zz"
                    and _same_qubits(node, succs[0])
                ):
                    nxt = succs[0]
                    if not _is_number(nxt.op.params[0]):
                        continue
                    theta = _mod1_turns(node.op.params[0] + nxt.op.params[0])
                    if _near(theta, 0.0):
                        dag.remove_op_node(nxt)
                        dag.remove_op_node(node)
                    else:
                        sub = DAGCircuit()
                        for qreg in dag.qregs.values():
                            sub.add_qreg(qreg)
                        sub.apply_operation_back(ZZGate(theta), list(nxt.qargs))
                        dag.substitute_node_with_dag(
                            nxt, sub, {q: q for q in nxt.qargs}
                        )
                        dag.remove_op_node(node)
                    changed = True
                    break
        return dag


class FuseConsecutiveMS(TransformationPass):
    """
    Fuse adjacent MS(φ0,φ1,θ1) then MS(φ0,φ1,θ2) on the same pair into MS(φ0,φ1,θ1+θ2).
    Only applies when φ0, φ1 match numerically (within tolerance).
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        while changed:
            changed = False
            for node in list(dag.topological_op_nodes()):
                if node.op.name != "ms" or not all(
                    _is_number(p) for p in node.op.params[:3]
                ):
                    continue
                phi0, phi1, theta1 = node.op.params[:3]
                qargs = node.qargs
                succs = _first_succ_ops_on_qubits(dag, node, qargs)
                if (
                    len(succs) == 2
                    and succs[0] is succs[1]
                    and succs[0].op.name == "ms"
                    and _same_qubits(node, succs[0])
                ):
                    nxt = succs[0]
                    if not all(_is_number(p) for p in nxt.op.params[:3]):
                        continue
                    phi0b, phi1b, theta2 = nxt.op.params[:3]
                    if not (_near(phi0, phi0b) and _near(phi1, phi1b)):
                        continue
                    theta = _mod1_turns(theta1 + theta2)
                    if _near(theta, 0.0):
                        dag.remove_op_node(nxt)
                        dag.remove_op_node(node)
                    else:
                        sub = DAGCircuit()
                        for qreg in dag.qregs.values():
                            sub.add_qreg(qreg)
                        sub.apply_operation_back(
                            MSGate(_mod1_turns(phi0), _mod1_turns(phi1), theta),
                            list(nxt.qargs),
                        )
                        dag.substitute_node_with_dag(
                            nxt, sub, {q: q for q in nxt.qargs}
                        )
                        dag.remove_op_node(node)
                    changed = True
                    break
        return dag


class ConjugateGPI2ByGPI(TransformationPass):
    """
    Collapse GPI(γ) • GPI2(α) • GPI(γ)  →  GPI2(2γ - α)  on the same wire.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        to_remove = []
        for mid in dag.topological_op_nodes():
            if mid.op.name != "gpi2" or not _is_number(mid.op.params[0]):
                continue
            preds = [
                p
                for p in dag.quantum_predecessors(mid)
                if isinstance(p, DAGOpNode) and (mid.qargs[0] in p.qargs)
            ]
            succs = [
                s
                for s in dag.quantum_successors(mid)
                if isinstance(s, DAGOpNode) and (mid.qargs[0] in s.qargs)
            ]
            if not preds or not succs:
                continue
            left, right = preds[-1], succs[0]
            if (
                left.op.name == "gpi"
                and right.op.name == "gpi"
                and _is_number(left.op.params[0])
                and _is_number(right.op.params[0])
                and _near(left.op.params[0], right.op.params[0])
            ):
                gamma = left.op.params[0]
                alpha = mid.op.params[0]
                new_phi = _mod1_turns(2 * gamma - alpha)
                sub = DAGCircuit()
                for qreg in dag.qregs.values():
                    sub.add_qreg(qreg)
                sub.apply_operation_back(GPI2Gate(new_phi), [mid.qargs[0]])
                dag.substitute_node_with_dag(mid, sub, {mid.qargs[0]: mid.qargs[0]})
                to_remove.extend([left, right])
        for n in to_remove:
            if n in dag.op_nodes():
                dag.remove_op_node(n)
        return dag


class CommuteGPI2AcrossGPI(TransformationPass):
    """
    Reorder adjacent GPI2(a) then GPI(b) on the same wire:
        GPI2(a) • GPI(b)  →  GPI(b) • GPI2(2b - a)
    This helps expose cancellations and shorten streaks after subsequent passes.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        while changed:
            changed = False
            for left in list(dag.topological_op_nodes()):
                if left.op.name != "gpi2" or not _is_number(left.op.params[0]):
                    continue
                succs = [
                    s
                    for s in dag.quantum_successors(left)
                    if isinstance(s, DAGOpNode) and (left.qargs[0] in s.qargs)
                ]
                if not succs:
                    continue
                right = succs[0]
                if right.op.name == "gpi" and _is_number(right.op.params[0]):
                    left_op = left.op.params[0]
                    right_op = right.op.params[0]
                    new_phi = _mod1_turns(2 * right_op - left_op)
                    sub = DAGCircuit()
                    for qreg in dag.qregs.values():
                        sub.add_qreg(qreg)
                    sub.apply_operation_back(GPIGate(right_op), [left.qargs[0]])
                    sub.apply_operation_back(GPI2Gate(new_phi), [left.qargs[0]])
                    dag.substitute_node_with_dag(
                        left, sub, {left.qargs[0]: left.qargs[0]}
                    )
                    if right in dag.op_nodes():
                        dag.remove_op_node(right)
                    changed = True
                    break
        return dag
