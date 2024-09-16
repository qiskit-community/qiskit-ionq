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

"""Test rewrite rules used by IonQ custom transpiler."""

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
from qiskit_ionq.ionq_custom_transpiler import IonQTranspiler

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
    "gates",
    [
        [("CXGate", None, [0, 1]), ("CXGate", None, [0, 2]), ("CXGate", None, [0, 3]), ("CXGate", None, [0, 4])],
        [("HGate", None, [0]), ("CHGate", None, [0, 1]), ("CHGate", None, [0, 2]), ("CHGate", None, [0, 3]), ("CHGate", None, [0, 4])],
        [("HGate", None, [0]), ("CHGate", None, [0, 1]), ("HGate", None, [1]), ("CHGate", None, [0, 2]), ("HGate", None, [2]), ("CHGate", None, [0, 3]), ("HGate", None, [3]), ("CHGate", None, [0, 4])],
        [("XGate", None, [0]), ("CHGate", None, [0, 1]), ("XGate", None, [1]), ("CHGate", None, [0, 2]), ("XGate", None, [2]), ("CHGate", None, [0, 3]), ("XGate", None, [3]), ("CHGate", None, [0, 4])],
        [("SGate", None, [0]), ("CHGate", None, [0, 1]), ("TGate", None, [1]), ("CHGate", None, [1, 2]), ("SGate", None, [2]), ("CHGate", None, [2, 3]), ("TGate", None, [3]), ("CHGate", None, [3, 4])],
        [("SGate", None, [0]), ("CHGate", None, [0, 1]), ("TGate", None, [1]), ("CHGate", None, [0, 2]), ("XGate", None, [2]), ("CHGate", None, [0, 3]), ("YGate", None, [3]), ("CHGate", None, [0, 4])],
    ],
    ids=lambda val: f"{val}",
)
def test_ionq_custom_transpiler(gates):
    # create a quantum circuit
    qr = QuantumRegister(5)
    circuit = QuantumCircuit(qr)
    for gate_name, param, qubits in gates:
        append_gate(circuit, gate_name, param, qubits)

    # transpile circuit to native gates
    provider = ionq_provider.IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
    transpiled_circuit_unoptimized = transpile(circuit, backend)

    # simulate the unoptimized circuit
    statevector_unoptimized = Statevector(transpiled_circuit_unoptimized)
    probabilities_unoptimized = np.abs(statevector_unoptimized) ** 2

    # optimized transpilation of circuit to native gates
    custom_transpiler = IonQTranspiler(backend)
    transpiled_circuit_optimized = custom_transpiler.transpile(circuit, optimization_level=1)

    # simulate the optimized circuit
    statevector_optimized = Statevector(transpiled_circuit_optimized)
    probabilities_optimized = np.abs(statevector_optimized) ** 2

    # print(transpiled_circuit_unoptimized)
    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # print(transpiled_circuit_optimized)

    np.testing.assert_allclose(
        probabilities_unoptimized,
        probabilities_optimized,
        atol=1e-3,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {circuit}"
        ),
    )