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

"""Final transpiler stage rewriting standard gates to IonQ native gates.

Native-gateset targets advertise standard-gate proxies (``r``/``rz`` plus
``rzz`` or ``rxx``) so the preset pipeline runs on Rust-native operations:
Python-defined gates in the optimization stage deadlock Qiskit >= 2.5.0's
multithreaded passes on the GIL.  The ``ionq_native`` scheduling-stage plugin
(selected via ``IonQBackend.get_scheduling_stage_plugin``) restores the
documented ``gpi``/``gpi2``/``ms``/``zz`` output basis, exactly.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager, TransformationPass, TranspilerError
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from .ionq_gates import GPIGate, GPI2Gate, MSGate, ZZGate

_TWO_PI = 2 * math.pi
_EPS = 1e-9


def _u_to_native(theta, phi, lam):
    """Gates and global phase with U(theta, phi, lam) == e^{i*phase} * gates.

    Symbolic-parameter safe.  The phase term makes the identity exact.
    """
    gates = [
        GPI2Gate(0.5 - lam / _TWO_PI),
        GPIGate((theta + phi - lam) / (2 * _TWO_PI)),
        GPI2Gate(0.5 + phi / _TWO_PI),
    ]
    return gates, (phi + lam) / 2 - math.pi / 2


def _native_1q_sequence(matrix):
    """Exactly synthesize a 2x2 unitary into 0-3 native gates plus a phase.

    Candidates are tried shortest-first and each is verified numerically, so a
    wrong candidate can never corrupt the circuit; the 3-gate GPI2.GPI.GPI2
    fallback is universal.
    """
    candidates = []
    a00 = abs(matrix[0, 0])
    if a00 > 1 - _EPS:  # diagonal: nothing, or two GPIs (no virtual RZ on IonQ)
        delta = cmath.phase(matrix[1, 1]) - cmath.phase(matrix[0, 0])
        candidates += [[], [GPIGate(0.0), GPIGate((delta / 2) / _TWO_PI)]]
    elif a00 < _EPS:  # anti-diagonal: a single GPI
        delta = cmath.phase(matrix[1, 0]) - cmath.phase(matrix[0, 1])
        candidates.append([GPIGate((delta / 2) / _TWO_PI)])
    elif abs(a00 - math.sqrt(0.5)) < _EPS:  # a single GPI2
        delta = cmath.phase(matrix[1, 0]) - cmath.phase(matrix[0, 0])
        candidates.append([GPI2Gate((delta + math.pi / 2) / _TWO_PI)])

    # Universal fallback: extract U-gate angles, apply the exact identity.
    if a00 < _EPS:
        gamma = cmath.phase(-matrix[0, 1])
        theta, phi, lam = math.pi, cmath.phase(matrix[1, 0]) - gamma, 0.0
    else:
        gamma = cmath.phase(matrix[0, 0])
        theta = 2 * math.atan2(abs(matrix[1, 0]), a00)
        if abs(matrix[1, 0]) < _EPS:
            phi, lam = 0.0, cmath.phase(matrix[1, 1]) - gamma
        else:
            phi = cmath.phase(matrix[1, 0]) - gamma
            lam = cmath.phase(-matrix[0, 1]) - gamma
    candidates.append(_u_to_native(theta, phi, lam)[0])

    for gates in candidates:
        composed = np.eye(2, dtype=complex)
        for gate in gates:
            composed = np.asarray(gate.to_matrix(), dtype=complex) @ composed
        ratio = matrix @ composed.conj().T
        if (
            abs(ratio[0, 1]) < _EPS
            and abs(ratio[1, 0]) < _EPS
            and abs(ratio[0, 0] - ratio[1, 1]) < _EPS
            and abs(abs(ratio[0, 0]) - 1) < _EPS
        ):
            return gates, cmath.phase(ratio[0, 0])
    raise TranspilerError("input matrix is not a 1-qubit unitary")


class ConvertToNativeGates(TransformationPass):
    """Rewrite an optimized standard-gate DAG onto IonQ native gates.

    ``rzz(t) -> zz(t/2pi)``, ``rxx(t) -> ms(0, 0, t/2pi)``; bound 1q runs are
    collapsed into at most three ``gpi``/``gpi2`` gates by exact synthesis;
    unbound ``r``/``rz`` rotations are mapped symbolically.  Gates that are
    already native are left untouched.
    """

    def run(self, dag):
        for node in dag.op_nodes():
            op = node.op
            if op.name == "rzz":
                dag.substitute_node(node, ZZGate(op.params[0] / _TWO_PI), inplace=True)
            elif op.name == "rxx":
                dag.substitute_node(node, MSGate(0.0, 0.0, op.params[0] / _TWO_PI), inplace=True)
            elif op.name in ("r", "rz") and op.is_parameterized():
                # Symbolic: r(t,p) == U(t, p-pi/2, pi/2-p); rz(l) == e^{-il/2} U(0,0,l)
                if op.name == "r":
                    gates, phase = _u_to_native(
                        op.params[0], op.params[1] - math.pi / 2, math.pi / 2 - op.params[1]
                    )
                else:
                    gates, phase = _u_to_native(0, 0, op.params[0])
                    phase = phase - op.params[0] / 2
                mini = QuantumCircuit(1, global_phase=phase)
                for gate in gates:
                    mini.append(gate, [0])
                dag.substitute_node_with_dag(node, circuit_to_dag(mini))

        for run in dag.collect_1q_runs():
            if {node.op.name for node in run} <= {"gpi", "gpi2"}:
                continue  # already native; leave user-written gates alone
            matrix = np.eye(2, dtype=complex)
            for node in run:
                matrix = np.asarray(node.op.to_matrix(), dtype=complex) @ matrix
            gates, phase = _native_1q_sequence(matrix)
            mini = QuantumCircuit(1, global_phase=phase)
            for gate in gates:
                mini.append(gate, [0])
            for node in run[1:]:
                dag.remove_op_node(node)
            dag.substitute_node_with_dag(run[0], circuit_to_dag(mini))
        return dag


class IonQNativeOutputPlugin(PassManagerStagePlugin):
    """``scheduling``-stage plugin (name ``ionq_native``) producing native output."""

    def pass_manager(self, pass_manager_config=None, optimization_level=None) -> PassManager:
        return PassManager([ConvertToNativeGates()])
