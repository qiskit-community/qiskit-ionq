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

Z rotations are virtual on IonQ hardware: :class:`ConvertToNativeGates` tracks
them as per-qubit phase frames instead of emitting physical gates.  A frame
``Rz(f)`` passes through ``zz`` unchanged, is absorbed into the phase
parameters of ``ms``, and folds into the synthesis of the next one-qubit run,
so interior runs cost at most two ``gpi``/``gpi2`` gates and diagonal runs
cost none.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from qiskit.transpiler import PassManager, TransformationPass, TranspilerError
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from .ionq_gates import GPIGate, GPI2Gate, MSGate, ZZGate

_TWO_PI = 2 * math.pi
_EPS = 1e-9


def _u_to_native(theta, phi, lam):
    """Gates and global phase with U(theta, phi, lam) == e^{i*phase} * gates.

    Symbolic-parameter safe.
    """
    gates = [
        GPI2Gate(0.5 - lam / _TWO_PI),
        GPIGate((theta + phi - lam) / (2 * _TWO_PI)),
        GPI2Gate(0.5 + phi / _TWO_PI),
    ]
    return gates, (phi + lam) / 2 - math.pi / 2


def _residual(matrix, gates):
    """Return ``(f, phase)`` with matrix == e^{i*phase} Rz(f) @ gates, or None."""
    composed = np.eye(2, dtype=complex)
    for gate in gates:
        composed = np.asarray(gate.to_matrix(), dtype=complex) @ composed
    res = matrix @ composed.conj().T
    if abs(res[0, 1]) > _EPS or abs(res[1, 0]) > _EPS or abs(abs(res[0, 0]) - 1) > _EPS:
        return None
    frame = cmath.phase(res[1, 1]) - cmath.phase(res[0, 0])
    frame = (
        frame + math.pi
    ) % _TWO_PI - math.pi  # Rz(f±2pi) = -Rz(f): sign joins the phase
    return frame, cmath.phase(res[0, 0] * cmath.exp(0.5j * frame))


def _flush_1q(matrix):
    """Synthesize ``matrix`` into <=2 native gates plus a residual Rz frame.

    Every candidate is verified numerically via :func:`_residual`; the
    two-GPI2 form (with a residual frame) is universal.
    """
    a00 = abs(matrix[0, 0])
    if a00 > 1 - _EPS:  # diagonal: pure frame
        candidates = [[]]
    elif a00 < _EPS:  # anti-diagonal
        x = (cmath.phase(matrix[1, 0]) - cmath.phase(matrix[0, 1])) / 2
        candidates = [[GPIGate(x / _TWO_PI)]]
    elif abs(a00 - math.sqrt(0.5)) < _EPS:
        x = -cmath.phase(1j * matrix[0, 1] / matrix[0, 0])
        candidates = [[GPI2Gate(x / _TWO_PI)]]
    else:
        delta = 2 * math.asin(min(a00, 1.0))
        frame = cmath.phase(matrix[1, 1]) - cmath.phase(matrix[0, 0]) - math.pi + delta
        total = cmath.phase(matrix[1, 0]) - cmath.phase(matrix[0, 1]) - frame
        candidates = [
            [
                GPI2Gate(((s + delta) / 2) / _TWO_PI),
                GPI2Gate(((s - delta) / 2) / _TWO_PI),
            ]
            for s in (total, total + _TWO_PI)  # phase wrap makes s ambiguous mod 2pi
        ]
    for gates in candidates:
        result = _residual(matrix, gates)
        if result is not None:
            return gates, result[0], result[1]
    raise TranspilerError("input matrix is not a 1-qubit unitary")


def _settle_1q(matrix):
    """Synthesize ``matrix`` into <=3 native gates with no residual frame."""
    a00 = abs(matrix[0, 0])
    candidates = []
    if a00 > 1 - _EPS:  # diagonal: identity, else two GPIs
        delta = cmath.phase(matrix[1, 1]) - cmath.phase(matrix[0, 0])
        candidates = [[], [GPIGate(0.0), GPIGate((delta / 2) / _TWO_PI)]]
    elif a00 < _EPS:  # anti-diagonal
        x = (cmath.phase(matrix[1, 0]) - cmath.phase(matrix[0, 1])) / 2
        candidates = [[GPIGate(x / _TWO_PI)]]
    elif abs(a00 - math.sqrt(0.5)) < _EPS:
        x = cmath.phase(matrix[1, 0]) - cmath.phase(matrix[0, 0]) + math.pi / 2
        candidates = [[GPI2Gate(x / _TWO_PI)]]
    # Universal fallback: extract U-gate angles, apply the exact identity.
    gamma = cmath.phase(matrix[0, 0])
    theta = 2 * math.atan2(abs(matrix[1, 0]), a00)
    if abs(matrix[1, 0]) < _EPS:
        phi, lam = 0.0, cmath.phase(matrix[1, 1]) - gamma
    else:
        phi = cmath.phase(matrix[1, 0]) - gamma
        lam = cmath.phase(-matrix[0, 1]) - gamma
    candidates.append(_u_to_native(theta, phi, lam)[0])
    for gates in candidates:
        result = _residual(matrix, gates)
        if result is not None and abs(result[0]) < _EPS:
            return gates, result[1]
    raise TranspilerError("input matrix is not a 1-qubit unitary")


def _bound_1q_matrix(op):
    if op.num_qubits != 1 or op.num_clbits or op.is_parameterized():
        return None
    try:
        return np.asarray(op.to_matrix(), dtype=complex)
    except Exception:  # pylint: disable=broad-except
        return None


class ConvertToNativeGates(TransformationPass):
    """Rewrite an optimized standard-gate DAG onto IonQ native gates.

    ``rzz(t) -> zz(t/2pi)``; ``rxx(t) -> ms(f0/2pi, f1/2pi, t/2pi)`` where the
    ``ms`` phases absorb the pending Rz frames; bound one-qubit runs collapse
    into at most two ``gpi``/``gpi2`` gates plus a frame update; unbound
    ``r``/``rz`` rotations are mapped symbolically.  Frames are materialized
    (two GPIs at most) at measurements, barriers, unhandled operations, and
    the end of the circuit, so the rewrite is exact including global phase.
    """

    def run(self, dag):
        new_dag = dag.copy_empty_like()
        pending = {}  # Qubit -> accumulated 2x2 unitary
        frames = {}  # Qubit -> pending virtual Rz angle (radians)

        def emit(qubit, gates, phase):
            for gate in gates:
                new_dag.apply_operation_back(gate, (qubit,), ())
            new_dag.global_phase = new_dag.global_phase + phase

        def with_frame(qubit):
            matrix = pending.pop(qubit, np.eye(2, dtype=complex))
            frame = frames.pop(qubit, 0.0)
            return matrix @ np.diag([np.exp(-0.5j * frame), np.exp(0.5j * frame)])

        def flush(qubit):  # synthesize pending gates, keep the residual as a frame
            if qubit in pending:
                gates, frame, phase = _flush_1q(with_frame(qubit))
                frames[qubit] = frame
                emit(qubit, gates, phase)

        def settle(qubit):  # synthesize everything, leaving no frame behind
            if qubit in pending or abs(frames.get(qubit, 0.0)) > _EPS:
                gates, phase = _settle_1q(with_frame(qubit))
                emit(qubit, gates, phase)

        for node in dag.topological_op_nodes():
            op, qargs = node.op, node.qargs
            matrix = _bound_1q_matrix(op)
            if matrix is not None:
                qubit = qargs[0]
                pending[qubit] = matrix @ pending.get(qubit, np.eye(2, dtype=complex))
            elif op.name == "rzz":  # frames commute with zz
                for qubit in qargs:
                    flush(qubit)
                new_dag.apply_operation_back(ZZGate(op.params[0] / _TWO_PI), qargs, ())
            elif op.name == "rxx":  # frames pass through unchanged as ms phases
                for qubit in qargs:
                    flush(qubit)
                phases = [-frames.get(qubit, 0.0) / _TWO_PI for qubit in qargs]
                new_dag.apply_operation_back(
                    MSGate(phases[0], phases[1], op.params[0] / _TWO_PI), qargs, ()
                )
            elif op.name in ("r", "rz") and op.is_parameterized():
                settle(qargs[0])
                if op.name == "r":
                    gates, phase = _u_to_native(
                        op.params[0],
                        op.params[1] - math.pi / 2,
                        math.pi / 2 - op.params[1],
                    )
                else:
                    gates, phase = _u_to_native(0, 0, op.params[0])
                    phase = phase - op.params[0] / 2
                emit(qargs[0], gates, phase)
            else:
                for qubit in qargs:
                    settle(qubit)
                new_dag.apply_operation_back(op, qargs, node.cargs)
        for qubit in dag.qubits:
            settle(qubit)
        return new_dag


class IonQNativeOutputPlugin(PassManagerStagePlugin):
    """``scheduling``-stage plugin (name ``ionq_native``) producing native output."""

    def pass_manager(
        self, pass_manager_config=None, optimization_level=None
    ) -> PassManager:
        return PassManager([ConvertToNativeGates()])
