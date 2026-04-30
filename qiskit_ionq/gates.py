"""IonQ native gate definitions for Qiskit.

Phase parameters (phi, phi0, phi1) are in turns (fractions of 2*pi).
Interaction parameters (angle) are in units of pi (0.25 = pi/4 radians).
"""

from __future__ import annotations

import numpy as np
from ionq_core import gpi2_matrix, gpi_matrix, ms_matrix, zz_matrix
from qiskit.circuit import Gate, QuantumCircuit


class _IonQGate(Gate):
    """Base class that provides a unitary-based _define() for all IonQ gates."""

    def _define(self):
        qubits = list(range(self.num_qubits))
        q = QuantumCircuit(self.num_qubits, name=self.name)
        q.unitary(self.to_matrix(), qubits)
        self.definition = q


class GPIGate(_IonQGate):
    """Single-qubit GPI gate: pi rotation about axis at angle phi in the XY plane."""

    def __init__(self, phi: float, *, label: str | None = None):
        super().__init__("gpi", 1, [phi], label=label)

    def __array__(self, dtype=None, copy=None):
        return np.array(gpi_matrix(float(self.params[0])), dtype=dtype)


class GPI2Gate(_IonQGate):
    """Single-qubit GPI2 gate: pi/2 rotation about axis at angle phi in the XY plane."""

    def __init__(self, phi: float, *, label: str | None = None):
        super().__init__("gpi2", 1, [phi], label=label)

    def __array__(self, dtype=None, copy=None):
        return np.array(gpi2_matrix(float(self.params[0])), dtype=dtype)


class MSGate(_IonQGate):
    """Two-qubit Molmer-Sorensen gate."""

    def __init__(
        self, phi0: float, phi1: float, angle: float = 0.25, *, label: str | None = None
    ):
        super().__init__("ms", 2, [phi0, phi1, angle], label=label)

    def __array__(self, dtype=None, copy=None):
        phi0, phi1, angle = (float(p) for p in self.params)
        return np.array(ms_matrix(phi0, phi1, angle), dtype=dtype)


class ZZGate(_IonQGate):
    """Two-qubit ZZ interaction gate."""

    def __init__(self, angle: float, *, label: str | None = None):
        super().__init__("zz", 2, [angle], label=label)

    def __array__(self, dtype=None, copy=None):
        return np.array(zz_matrix(float(self.params[0])), dtype=dtype)
