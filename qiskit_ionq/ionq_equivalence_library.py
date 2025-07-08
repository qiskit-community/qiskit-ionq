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
from qiskit.circuit.library import CXGate, RXGate, RZGate, UGate, XGate, CU3Gate

from qiskit_ionq.exceptions import IonQBackendNotSupportedError
from .ionq_gates import GPIGate, GPI2Gate, MSGate, ZZGate


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


def cx_gate_equivalence_ms() -> None:
    """Add MS gate based CX gate equivalence to the SessionEquivalenceLibrary."""
    q = QuantumRegister(2, "q")
    cx_gate = QuantumCircuit(q)
    cx_gate.append(GPI2Gate(1 / 4), [0])
    cx_gate.append(MSGate(0, 0), [0, 1])
    cx_gate.append(GPI2Gate(1 / 2), [0])
    cx_gate.append(GPI2Gate(1 / 2), [1])
    cx_gate.append(GPI2Gate(-1 / 4), [0])
    if SessionEquivalenceLibrary.has_entry(CXGate()):
        SessionEquivalenceLibrary.set_entry(CXGate(), [cx_gate])
    else:
        SessionEquivalenceLibrary.add_equivalence(CXGate(), cx_gate)


def cx_gate_equivalence_zz() -> None:
    """Add ZZ gate based CX gate equivalence to the SessionEquivalenceLibrary.
    q_0: ────■────Sdag──────
             │ZZ
    q_1: H───■────Sdag────H─
    """
    q = QuantumRegister(2, "q")
    cx_gate = QuantumCircuit(q)
    # H
    cx_gate.append(GPI2Gate(0), [1])
    cx_gate.append(GPIGate(-0.125), [1])
    cx_gate.append(GPI2Gate(0.5), [1])
    # ZZ
    cx_gate.append(ZZGate(), [0, 1])
    # Sdag
    cx_gate.append(GPI2Gate(0.75), [0])
    cx_gate.append(GPIGate(0.125), [0])
    cx_gate.append(GPI2Gate(0.5), [0])
    #  H * Sdag
    cx_gate.append(GPI2Gate(1.25), [1])
    cx_gate.append(GPIGate(0.5), [1])
    cx_gate.append(GPI2Gate(0.5), [1])
    if SessionEquivalenceLibrary.has_entry(CXGate()):
        SessionEquivalenceLibrary.set_entry(CXGate(), [cx_gate])
    else:
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


def zz_gate_equivalence() -> None:
    """Add ZZ gate equivalence to the SessionEquivalenceLibrary."""
    q = QuantumRegister(2, "q")
    zz_gate = QuantumCircuit(q)
    zz_gate.h(1)
    zz_gate.append(CXGate(), [0, 1])
    zz_gate.s(0)
    zz_gate.h(1)
    zz_gate.s(1)
    SessionEquivalenceLibrary.add_equivalence(ZZGate(), zz_gate)


def add_equivalences(backend_name, noise_model=None) -> None:
    """Add IonQ gate equivalences to the SessionEquivalenceLibrary."""
    u_gate_equivalence()
    if backend_name in (
        "ionq_mock_backend",
        "ionq_qpu",
        "ionq_qpu.harmony",
        "ionq_qpu.aria-1",
        "ionq_qpu.aria-2",
    ):
        cx_gate_equivalence_ms()
    elif backend_name in (
        "ionq_qpu.forte-1",
        "ionq_qpu.forte-enterprise-1",
        "ionq_qpu.forte-enterprise-2",
    ):
        cx_gate_equivalence_zz()
    elif backend_name == "ionq_simulator":
        if noise_model is None or noise_model in [
            "harmony",
            "harmony-1",
            "harmony-2",
            "aria-1",
            "aria-2",
            "ideal",
            "ideal-sampled",
        ]:
            cx_gate_equivalence_ms()
        elif noise_model in ["forte-1", "forte-enterprise-1", "forte-enterprise-2"]:
            cx_gate_equivalence_zz()
    else:
        raise IonQBackendNotSupportedError(
            f"The backend with name {backend_name} is not supported. "
            "The following backends names are supported: simulator or ionq_simulator "
            "(with noise models: ideal as default, ideal-sampled, aria-1, aria-2, forte-1, "
            "forte-enterprise-1, forte-enterprise-2, and legacy harmony, harmony-1, harmony-2) "
            "qpu.aria-1 or ionq_qpu.aria-1, qpu.aria-2 or ionq_qpu.aria-2, "
            "qpu.forte-1 or ionq_qpu.forte-1, "
            "qpu.forte-enterprise-1 or ionq_qpu.forte-enterprise-1, "
            "qpu.forte-enterprise-2 or ionq_qpu.forte-enterprise-2."
        )
    gpi_gate_equivalence()
    gpi2_gate_equivalence()
    ms_gate_equivalence()
    zz_gate_equivalence()
