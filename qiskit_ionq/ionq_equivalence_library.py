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


"""Equivalences for IonQ native gates."""

import numpy as np
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit import QuantumRegister, QuantumCircuit, Parameter
from qiskit.circuit.library import (
    XGate,
    CXGate,
    RXGate,
    RZGate,
    RZZGate,
    UGate,
)
from .ionq_gates import GPIGate, GPI2Gate, MSGate, ZZGate


# -------- 1q gates --------


def u_gate_equivalence() -> None:
    """U(θ,φ,λ) -> GPI2(1/2 - λ/2π) * GPI(θ/4π + φ/4π - λ/4π) * GPI2(1/2 + φ/2π)."""
    q = QuantumRegister(1, "q")
    theta = Parameter("theta_param")
    phi = Parameter("phi_param")
    lam = Parameter("lambda_param")

    circ = QuantumCircuit(q)
    circ.append(GPI2Gate(0.5 - lam / (2 * np.pi)), [0])
    circ.append(
        GPIGate(theta / (4 * np.pi) + phi / (4 * np.pi) - lam / (4 * np.pi)), [0]
    )
    circ.append(GPI2Gate(0.5 + phi / (2 * np.pi)), [0])
    SessionEquivalenceLibrary.add_equivalence(UGate(theta, phi, lam), circ)


def gpi_gate_equivalence() -> None:
    """GPI(φ) -> X * RZ(4πφ) (for Aer/QIS simulation)."""
    q = QuantumRegister(1, "q")
    phi = Parameter("phi_param")
    circ = QuantumCircuit(q)
    circ.append(XGate(), [0])
    circ.append(RZGate(4 * phi * np.pi), [0])
    SessionEquivalenceLibrary.add_equivalence(GPIGate(phi), circ)


def gpi2_gate_equivalence() -> None:
    """GPI2(φ) -> RZ(-2πφ) * RX(π/2) * RZ(2πφ) (for Aer/QIS simulation)."""
    q = QuantumRegister(1, "q")
    phi = Parameter("phi_param")
    circ = QuantumCircuit(q)
    circ.append(RZGate(-2 * phi * np.pi), [0])
    circ.append(RXGate(np.pi / 2), [0])
    circ.append(RZGate(2 * phi * np.pi), [0])
    SessionEquivalenceLibrary.add_equivalence(GPI2Gate(phi), circ)


# -------- 2q native gates -> standard rotations (helps simulation & pattern matching) --------


def zz_gate_equivalence() -> None:
    """ZZ(θ) -> RZZ(2πθ)."""
    q = QuantumRegister(2, "q")
    theta = Parameter("theta_param")
    circ = QuantumCircuit(q)
    circ.append(RZZGate(2 * np.pi * theta), [0, 1])
    SessionEquivalenceLibrary.add_equivalence(ZZGate(theta), circ)


# -------- CX constructions (one for MS backends, one for Forte/ZZ) --------


def cx_gate_equivalence_via_ms() -> None:
    """CX via one MS(0,0,1/4) and three GPI2 rotations (IonQ MS-native)."""
    q = QuantumRegister(2, "q")
    circ = QuantumCircuit(q)
    circ.append(GPI2Gate(1 / 4), [0])
    circ.append(MSGate(0, 0, 1 / 4), [0, 1])
    circ.append(GPI2Gate(1 / 2), [0])
    circ.append(GPI2Gate(1 / 2), [1])
    circ.append(GPI2Gate(-1 / 4), [0])
    SessionEquivalenceLibrary.add_equivalence(CXGate(), circ)


def cx_gate_equivalence_via_zz() -> None:
    """CX via ZZ(1/4): H_t * ZZ(1/4) * S†_c * S†_t * H_t (IonQ Forte-native)."""
    q = QuantumRegister(2, "q")
    circ = QuantumCircuit(q)
    circ.h(1)
    circ.append(ZZGate(1 / 4), [0, 1])
    circ.sdg(0)
    circ.sdg(1)
    circ.h(1)
    SessionEquivalenceLibrary.add_equivalence(CXGate(), circ)


def add_equivalences() -> None:
    """Register all IonQ gate equivalences in the session library."""
    # 1q
    u_gate_equivalence()
    gpi_gate_equivalence()
    gpi2_gate_equivalence()
    # 2q
    zz_gate_equivalence()
    # CX (both backends)
    cx_gate_equivalence_via_ms()
    cx_gate_equivalence_via_zz()
