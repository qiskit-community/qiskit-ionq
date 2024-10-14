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

"""Equivalences for IonQ native gates."""

import numpy as np
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit import QuantumRegister, QuantumCircuit, Parameter
from qiskit.circuit.library import CXGate, RXGate, RZGate, UGate, XGate, CU3Gate
from .ionq_gates import GPIGate, GPI2Gate, MSGate


def u_gate_equivalence() -> None:
    """Add U gate equivalence to the SessionEquivalenceLibrary."""
    q = QuantumRegister(1, "q")
    theta_param = Parameter("theta_param")
    phi_param = Parameter("phi_param")
    lambda_param = Parameter("lambda_param")
    u_gate = QuantumCircuit(q)
    # this sequence can be compacted if virtual-z gates will be introduced
    u_gate.append(GPI2Gate(0.5 - lambda_param / (2 * np.pi)), [0])
    u_gate.append(
        GPIGate(
            theta_param / (4 * np.pi)
            + phi_param / (4 * np.pi)
            - lambda_param / (4 * np.pi)
        ),
        [0],
    )
    u_gate.append(GPI2Gate(0.5 + phi_param / (2 * np.pi)), [0])
    SessionEquivalenceLibrary.add_equivalence(
        UGate(theta_param, phi_param, lambda_param), u_gate
    )


def cx_gate_equivalence() -> None:
    """Add CX gate equivalence to the SessionEquivalenceLibrary."""
    q = QuantumRegister(2, "q")
    cx_gate = QuantumCircuit(q)
    cx_gate.append(GPI2Gate(1 / 4), [0])
    cx_gate.append(MSGate(0, 0), [0, 1])
    cx_gate.append(GPI2Gate(1 / 2), [0])
    cx_gate.append(GPI2Gate(1 / 2), [1])
    cx_gate.append(GPI2Gate(-1 / 4), [0])
    SessionEquivalenceLibrary.add_equivalence(CXGate(), cx_gate)


# Below are the rules needed for Aer simulator to simulate circuits containing IonQ native gates


def gpi_gate_equivalence() -> None:
    """Add GPI gate equivalence to the SessionEquivalenceLibrary."""
    q = QuantumRegister(1, "q")
    phi_param = Parameter("phi_param")
    gpi_gate = QuantumCircuit(q)
    gpi_gate.append(XGate(), [0])
    gpi_gate.append(RZGate(4 * phi_param * np.pi), [0])
    SessionEquivalenceLibrary.add_equivalence(GPIGate(phi_param), gpi_gate)


def gpi2_gate_equivalence() -> None:
    """Add GPI2 gate equivalence to the SessionEquivalenceLibrary."""
    q = QuantumRegister(1, "q")
    phi_param = Parameter("phi_param")
    gpi2_gate = QuantumCircuit(q)
    gpi2_gate.append(RZGate(-2 * phi_param * np.pi), [0])
    gpi2_gate.append(RXGate(np.pi / 2), [0])
    gpi2_gate.append(RZGate(2 * phi_param * np.pi), [0])
    SessionEquivalenceLibrary.add_equivalence(GPI2Gate(phi_param), gpi2_gate)


def ms_gate_equivalence() -> None:
    """Add MS gate equivalence to the SessionEquivalenceLibrary."""
    q = QuantumRegister(2, "q")
    phi0_param = Parameter("phi0_param")
    phi1_param = Parameter("phi1_param")
    theta_param = Parameter("theta_param")
    alpha_param = phi0_param + phi1_param
    beta_param = phi0_param - phi1_param
    ms_gate = QuantumCircuit(q)
    ms_gate.append(CXGate(), [1, 0])
    ms_gate.x(0)
    ms_gate.append(
        CU3Gate(
            2 * theta_param * np.pi,
            2 * alpha_param * np.pi - np.pi / 2,
            np.pi / 2 - 2 * alpha_param * np.pi,
        ),
        [0, 1],
    )
    ms_gate.x(0)
    ms_gate.append(
        CU3Gate(
            2 * theta_param * np.pi,
            -2 * beta_param * np.pi - np.pi / 2,
            np.pi / 2 + 2 * beta_param * np.pi,
        ),
        [0, 1],
    )
    ms_gate.append(CXGate(), [1, 0])
    SessionEquivalenceLibrary.add_equivalence(
        MSGate(phi0_param, phi1_param, theta_param), ms_gate
    )


def add_equivalences() -> None:
    """Add IonQ gate equivalences to the SessionEquivalenceLibrary."""
    u_gate_equivalence()
    cx_gate_equivalence()
    gpi_gate_equivalence()
    gpi2_gate_equivalence()
    ms_gate_equivalence()
