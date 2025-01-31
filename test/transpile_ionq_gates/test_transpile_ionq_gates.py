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

"""Test transpilation to native gatesets."""

import collections.abc
import numpy as np
import pytest

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    transpile,
)
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    HGate,
    IGate,
    PhaseGate,
    RGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    TGate,
    TdgGate,
    UGate,
    U1Gate,
    U2Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
    CHGate,
    CPhaseGate,
    CRXGate,
    RXXGate,
    CRYGate,
    RYYGate,
    CRZGate,
    RZZGate,
    RZXGate,
    XXMinusYYGate,
    XXPlusYYGate,
    ECRGate,
    CSGate,
    CSdgGate,
    SwapGate,
    iSwapGate,
    DCXGate,
    CUGate,
    CU1Gate,
    CU3Gate,
    CXGate,
    CYGate,
    CZGate,
)
from qiskit_ionq import ionq_provider

# Mapping from gate names to gate classes
gate_map = {
    # single-qubit gates
    "HGate": HGate,
    "IGate": IGate,
    "PhaseGate": PhaseGate,
    "RGate": RGate,
    "RXGate": RXGate,
    "RYGate": RYGate,
    "RZGate": RZGate,
    "SGate": SGate,
    "SdgGate": SdgGate,
    "SXGate": SXGate,
    "SXdgGate": SXdgGate,
    "TGate": TGate,
    "TdgGate": TdgGate,
    "UGate": UGate,
    "U1Gate": U1Gate,
    "U2Gate": U2Gate,
    "U3Gate": U3Gate,
    "XGate": XGate,
    "YGate": YGate,
    "ZGate": ZGate,
    # multi-qubit gates
    "CHGate": CHGate,
    "CPhaseGate": CPhaseGate,
    "CRXGate": CRXGate,
    "RXXGate": RXXGate,
    "CRYGate": CRYGate,
    "RYYGate": RYYGate,
    "CRZGate": CRZGate,
    "RZZGate": RZZGate,
    "RZXGate": RZXGate,
    "XXMinusYYGate": XXMinusYYGate,
    "XXPlusYYGate": XXPlusYYGate,
    "ECRGate": ECRGate,
    "CSGate": CSGate,
    "CSdgGate": CSdgGate,
    "SwapGate": SwapGate,
    "iSwapGate": iSwapGate,
    "DCXGate": DCXGate,
    "CUGate": CUGate,
    "CU1Gate": CU1Gate,
    "CU3Gate": CU3Gate,
    "CXGate": CXGate,
    "CYGate": CYGate,
    "CZGate": CZGate,
}


def append_gate(circuit, gate_name, param, qubits):
    """Append a gate to a circuit."""
    gate_class = gate_map[gate_name]
    if param is not None:
        if isinstance(param, collections.abc.Sequence):
            circuit.append(gate_class(*param), qubits)
        else:
            circuit.append(gate_class(param), qubits)
    else:
        circuit.append(gate_class(), qubits)


@pytest.mark.parametrize(
    "ideal_results, gates",
    [
        # single-qubit gates
        ([0.5, 0.5], [("HGate", None)]),
        ([1, 0], [("IGate", None)]),
        ([1, 0], [("PhaseGate", 0.25)]),
        ([0.984, 0.016], [("RGate", [0.25, 0.5])]),
        ([0.984, 0.016], [("RXGate", 0.25)]),
        ([0.984, 0.016], [("RYGate", 0.25)]),
        ([1, 0], [("RZGate", 0.25)]),
        ([1, 0], [("SGate", None)]),
        ([1, 0], [("SdgGate", None)]),
        ([0.5, 0.5], [("SXGate", None)]),
        ([0.5, 0.5], [("SXdgGate", None)]),
        ([1, 0], [("TGate", None)]),
        ([1, 0], [("TdgGate", None)]),
        ([0.984, 0.016], [("UGate", [0.25, 0.5, 0.75])]),
        ([1, 0], [("U1Gate", 0.25)]),
        ([0.5, 0.5], [("U2Gate", [0.25, 0.5])]),
        ([0.984, 0.016], [("U3Gate", [0.25, 0.5, 0.75])]),
        ([0, 1], [("XGate", None)]),
        ([0, 1], [("YGate", None)]),
        ([1, 0], [("ZGate", None)]),
        # sequence of single-qubit gates
        (
            [0.966, 0.034],
            [
                ("HGate", None),
                ("IGate", None),
                ("PhaseGate", 0.25),
                ("RGate", [0.25, 0.5]),
                ("RXGate", 0.25),
                ("RYGate", 0.25),
                ("RZGate", 0.25),
                ("SGate", None),
                ("SdgGate", None),
                ("SXGate", None),
                ("SXdgGate", None),
                ("TGate", None),
                ("TdgGate", None),
                ("UGate", [0.25, 0.5, 0.75]),
                ("U1Gate", 0.25),
                ("U2Gate", [0.25, 0.5]),
                ("U3Gate", [0.25, 0.5, 0.75]),
                ("XGate", None),
                ("YGate", None),
                ("ZGate", None),
            ],
        ),
    ],
    ids=lambda val: f"{val}",
)
def test_single_qubit_transpilation(ideal_results, gates):
    """Test transpiling single-qubit circuits to native gates."""
    # create a quantum circuit
    qr = QuantumRegister(1)
    circuit = QuantumCircuit(qr)
    for gate_name, param in gates:
        append_gate(circuit, gate_name, param, [0])

    # Transpile circuit to native gates using default simulator
    provider = ionq_provider.IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
    transpiled_circuit = transpile(circuit, backend)

    # simulate the circuit
    statevector = Statevector(transpiled_circuit)
    probabilities = np.abs(statevector) ** 2
    np.testing.assert_allclose(
        probabilities,
        ideal_results,
        atol=1e-3,
        err_msg=(
            f"Ideal: {np.round(ideal_results, 3)},\n"
            f"Actual: {np.round(probabilities, 3)},\n"
            f"Circuit: {circuit}"
        ),
    )

    # Transpile circuit to native gates. Transpiling to one qubit gates using forte should
    # make no difference w.r.t using default simulator, we test this scenario nevertheless.
    provider = ionq_provider.IonQProvider()
    backend = provider.get_backend("ionq_simulator_forte", gateset="native")
    transpiled_circuit = transpile(circuit, backend)

    # simulate the circuit
    statevector = Statevector(transpiled_circuit)
    probabilities = np.abs(statevector) ** 2
    np.testing.assert_allclose(
        probabilities,
        ideal_results,
        atol=1e-3,
        err_msg=(
            f"Ideal: {np.round(ideal_results, 3)},\n"
            f"Actual: {np.round(probabilities, 3)},\n"
            f"Circuit: {circuit}"
        ),
    )


@pytest.mark.parametrize(
    "ideal_results, gates",
    [
        # two-qubit gates
        (
            [0.984, 0.008, 0, 0.008],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CHGate", None, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CPhaseGate", 0.25, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CRXGate", 0.25, [0, 1])],
        ),
        (
            [0.969, 0.015, 0, 0.015],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("RXXGate", 0.25, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CRYGate", 0.25, [0, 1])],
        ),
        (
            [0.969, 0.015, 0, 0.015],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("RYYGate", 0.25, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CRZGate", 0.25, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("RZZGate", 0.25, [0, 1])],
        ),
        (
            [0.969, 0.015, 0.015, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("RZXGate", 0.25, [0, 1])],
        ),
        (
            [0.969, 0.016, 0, 0.015],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("XXMinusYYGate", 0.25, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("XXPlusYYGate", 0.25, [0, 1])],
        ),
        (
            [0.008, 0.492, 0.008, 0.492],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("ECRGate", None, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CSGate", None, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CSdgGate", None, [0, 1])],
        ),
        (
            [0.984, 0, 0.016, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("SwapGate", None, [0, 1])],
        ),
        (
            [0.984, 0, 0.016, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("iSwapGate", None, [0, 1])],
        ),
        (
            [0.984, 0, 0.016, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("DCXGate", None, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [
                ("U3Gate", [0.25, 0.5, 0.75], [0]),
                ("CUGate", [0.25, 0.5, 0.75, 1], [0, 1]),
            ],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CU1Gate", 0.25, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [
                ("U3Gate", [0.25, 0.5, 0.75], [0]),
                ("CU3Gate", [0.25, 0.5, 0.75], [0, 1]),
            ],
        ),
        (
            [0.984, 0, 0, 0.016],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CXGate", None, [0, 1])],
        ),
        (
            [0.984, 0, 0, 0.016],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CYGate", None, [0, 1])],
        ),
        (
            [0.984, 0.016, 0, 0],
            [("U3Gate", [0.25, 0.5, 0.75], [0]), ("CZGate", None, [0, 1])],
        ),
        # sequence of two-qubit gates
        (
            [0.012, 0.619, 0.350, 0.019],
            [
                ("U3Gate", [0.25, 0.5, 0.75], [0]),
                ("CHGate", None, [0, 1]),
                ("CPhaseGate", 0.25, [0, 1]),
                ("CRXGate", 0.25, [0, 1]),
                ("RXXGate", 0.25, [0, 1]),
                ("CRYGate", 0.25, [0, 1]),
                ("RYYGate", 0.25, [0, 1]),
                ("CRZGate", 0.25, [0, 1]),
                ("RZZGate", 0.25, [0, 1]),
                ("RZXGate", 0.25, [0, 1]),
                ("XXMinusYYGate", 0.25, [0, 1]),
                ("XXPlusYYGate", 0.25, [0, 1]),
                ("ECRGate", None, [0, 1]),
                ("CSGate", None, [0, 1]),
                ("CSdgGate", None, [0, 1]),
                ("SwapGate", None, [0, 1]),
                ("iSwapGate", None, [0, 1]),
                ("DCXGate", None, [0, 1]),
                ("CUGate", [0.25, 0.5, 0.75, 1], [0, 1]),
                ("CU1Gate", 0.25, [0, 1]),
                ("CU3Gate", [0.25, 0.5, 0.75], [0, 1]),
                ("CXGate", None, [0, 1]),
                ("CYGate", None, [0, 1]),
                ("CZGate", None, [0, 1]),
            ],
        ),
    ],
    ids=lambda val: f"{val}",
)
def test_two_qubit_transpilation(ideal_results, gates):
    """Test transpiling two-qubit circuits to native gates."""
    # create a quantum circuit
    qr = QuantumRegister(2)
    circuit = QuantumCircuit(qr)
    for gate_name, param, qubits in gates:
        append_gate(circuit, gate_name, param, qubits)

    # Transpile circuit to native gates using default simulator
    provider = ionq_provider.IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
    # Using optmization level 0 below is important here because ElidePermutations transpiler pass
    # in Qiskit will remove swap gates and instead premute qubits if optimization level is 2 or 3.
    # In the future this feature could be extended to optimization level 1 as well.
    transpiled_circuit = transpile(circuit, backend, optimization_level=0)

    # simulate the circuit
    statevector = Statevector(transpiled_circuit)
    probabilities = np.abs(statevector) ** 2
    np.testing.assert_allclose(
        probabilities,
        ideal_results,
        atol=1e-3,
        err_msg=(
            f"Ideal: {np.round(ideal_results, 3)},\n"
            f"Actual: {np.round(probabilities, 3)},\n"
            f"Circuit: {circuit}"
        ),
    )
    # Transpile circuit to native gates using forte
    provider = ionq_provider.IonQProvider()
    backend = provider.get_backend("ionq_simulator_forte", gateset="native")
    # Using optmization level 0 below is important here because ElidePermutations transpiler pass
    # in Qiskit will remove swap gates and instead premute qubits if optimization level is 2 or 3.
    # In the future this feature could be extended to optimization level 1 as well.
    transpiled_circuit = transpile(circuit, backend, optimization_level=0)

    # simulate the circuit
    statevector = Statevector(transpiled_circuit)
    probabilities = np.abs(statevector) ** 2
    np.testing.assert_allclose(
        probabilities,
        ideal_results,
        atol=1e-3,
        err_msg=(
            f"Ideal: {np.round(ideal_results, 3)},\n"
            f"Actual: {np.round(probabilities, 3)},\n"
            f"Circuit: {circuit}"
        ),
    )
