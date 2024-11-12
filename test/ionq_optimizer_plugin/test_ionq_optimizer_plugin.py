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
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManagerConfig
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
from qiskit_ionq import GPIGate, GPI2Gate, MSGate
from qiskit_ionq import (
    IonQProvider,
    TrappedIonOptimizerPlugin,
    TrappedIonOptimizerPluginSimpleRules,
    TrappedIonOptimizerPluginCompactGates,
)

# Mapping from gate names to gate classes
gate_map = {
    # IonQ gates
    "GPIGate": GPIGate,
    "GPI2Gate": GPI2Gate,
    "MSGate": MSGate,
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
    "gates, optimized_depth",
    [
        # GPI2(phi) * GPI2(phi + 0.5) = Id
        ([("GPI2Gate", [1], [0]), ("GPI2Gate", [1.5], [0])], 0),
        # GPI2(phi1) * GPI2(phi2) != Id  if phi1 + 0.5 is close but not phi2
        ([("GPI2Gate", [0], [0]), ("GPI2Gate", [0.501], [0])], 2),
        # GPI2(phi + 0.5) * GPI2(phi) = Id
        ([("GPI2Gate", [1.7], [0]), ("GPI2Gate", [1.2], [0])], 0),
        # GPI2(phi1) * GPI2(phi2) != Id if phi2 + 0.5 is close but not phi1
        ([("GPI2Gate", [1.701], [0]), ("GPI2Gate", [1.2], [0])], 2),
        # GPI2(phi) * GPI2(phi + 0.5) = Id
        (
            [
                ("GPI2Gate", [0.2], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [1], [0]),
            ],
            1,
        ),
        # GPI2(phi) * GPI2(phi + 0.5) = Id if phi2 + 0.5 is close but not phi1
        (
            [
                ("GPI2Gate", [0.19], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [1], [0]),
            ],
            3,
        ),
        # GPI(phi) * GPI(phi) = Id
        ([("GPIGate", [1], [0]), ("GPIGate", [1], [0])], 0),
        # GPI(phi1) * GPI(phi2) != Id if phi1 is close to phi2 but not identical
        ([("GPIGate", [1], [0]), ("GPIGate", [1.01], [0])], 2),
        # GPI(phi) * GPI(phi) = Id
        ([("GPIGate", [1], [0]), ("GPIGate", [1], [0]), ("GPIGate", [1], [0])], 1),
        # GPI(phi1) * GPI(phi2) != Id if phi1 is close to phi2 but not identical
        ([("GPIGate", [1], [0]), ("GPIGate", [1.01], [0]), ("GPIGate", [1], [0])], 3),
        # Four GPI2 with equal arguments should cancel
        (
            [
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [0]),
            ],
            0,
        ),
        # Four GPI2 with almost equal but not equal arguments should not cancel
        (
            [
                ("GPI2Gate", [0.701], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.699], [0]),
                ("GPI2Gate", [0.7], [0]),
            ],
            4,
        ),
        # Four GPI2 with equal arguments should cancel
        (
            [
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [0]),
            ],
            1,
        ),
        # Four GPI2 with almost equal but not equal arguments should not cancel
        (
            [
                ("GPI2Gate", [0.701], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.699], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.701], [0]),
            ],
            5,
        ),
        # Two GPI2 with equal arguments equal GPI
        ([("GPI2Gate", [1.7], [0]), ("GPI2Gate", [1.7], [0])], 1),
        # Two GPI2 with almost equal arguments do not change
        ([("GPI2Gate", [1.701], [0]), ("GPI2Gate", [1.7], [0])], 2),
        # Two GPI2 with equal arguments equal GPI
        (
            [
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
            ],
            2,
        ),
        # GPI * GPI2 * GPI2 -> GPI * GPI -> Id
        (
            [
                ("GPIGate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
            ],
            0,
        ),
        # GPI2 phi1 almost but not exactly GPI2 phi2
        (
            [
                ("GPIGate", [1.7], [0]),
                ("GPI2Gate", [1.701], [0]),
                ("GPI2Gate", [1.7], [0]),
            ],
            3,
        ),
        # GPI2 phi1 almost but not exactly GPI2 phi2
        (
            [
                ("GPI2Gate", [1.1], [0]),
                ("GPI2Gate", [1.101], [0]),
                ("GPIGate", [1.1], [0]),
            ],
            3,
        ),
        # GPI * GPI2 * GPI2 * (GPI2 * GPI2-dag) -> GPI * GPI * (GPI2 * GPI2-dag) -> * (GPI2 * GPI2-dag) -> Id
        (
            [
                ("GPIGate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.2], [0]),
            ],
            0,
        ),
        # GPI2-dag * GPI * GPI2 * GPI2 * GPI2  -> GPI2-dag * GPI * GPI * GPI2 -> GPI2-dag * GPI2 -> Id
        (
            [
                ("GPI2Gate", [2.2], [0]),
                ("GPIGate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
            ],
            0,
        ),
    ],
    ids=lambda val: f"{val}",
)
def test_ionq_optmizer_plugin_simple_one_qubit_rules(gates, optimized_depth):

    custom_pass_manager_plugin = TrappedIonOptimizerPluginSimpleRules()
    pass_manager_config = PassManagerConfig()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        pass_manager_config,
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(1)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("simulator", gateset="native")
    transpiled_circuit_unoptimized = transpile(
        qc, backend=backend, optimization_level=3
    )

    # simulate the unoptimized circuit
    statevector_unoptimized = Statevector.from_instruction(
        transpiled_circuit_unoptimized
    )
    probabilities_unoptimized = np.abs(statevector_unoptimized.data) ** 2

    # optimized transpilation of circuit to native gates
    optimized_circuit = custom_pass_manager.run(transpiled_circuit_unoptimized)

    # simulate the optimized circuit
    statevector_optimized = Statevector.from_instruction(optimized_circuit)
    probabilities_optimized = np.abs(statevector_optimized.data) ** 2

    np.testing.assert_allclose(
        probabilities_unoptimized,
        probabilities_optimized,
        atol=1e-3,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )

    optimized_dag = circuit_to_dag(optimized_circuit)
    assert optimized_dag.depth() == optimized_depth


@pytest.mark.parametrize(
    "gates, optimized_depth",
    [
        (
            [
                ("GPIGate", [2.2], [0]),
                ("GPIGate", [0.7], [0]),
                ("GPIGate", [1.7], [0]),
                ("GPIGate", [1.7], [0]),
            ],
            3,
        ),
        (
            [
                ("GPIGate", [1.8], [0]),
                ("GPIGate", [0.7], [0]),
                ("GPIGate", [1.1], [0]),
                ("GPIGate", [1.7], [0]),
                ("GPIGate", [2.7], [0]),
            ],
            3,
        ),
        (
            [
                ("GPIGate", [0.8], [0]),
                ("GPIGate", [2.7], [0]),
                ("GPIGate", [0.1], [0]),
                ("GPIGate", [2.7], [0]),
                ("GPIGate", [0.7], [0]),
                ("GPIGate", [1.7], [0]),
            ],
            3,
        ),
        (
            [
                ("GPI2Gate", [0.2], [0]),
                ("GPIGate", [0.7], [0]),
                ("GPI2Gate", [1.5], [0]),
                ("GPIGate", [2.8], [0]),
            ],
            3,
        ),
    ],
    ids=lambda val: f"{val}",
)
def test_ionq_optmizer_plugin_compact_more_than_three_gates(gates, optimized_depth):

    custom_pass_manager_plugin = TrappedIonOptimizerPluginCompactGates()
    pass_manager_config = PassManagerConfig()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        pass_manager_config,
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(1)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("simulator", gateset="native")
    transpiled_circuit_unoptimized = transpile(
        qc, backend=backend, optimization_level=3
    )

    # simulate the unoptimized circuit
    statevector_unoptimized = Statevector.from_instruction(
        transpiled_circuit_unoptimized
    )
    probabilities_unoptimized = np.abs(statevector_unoptimized.data) ** 2

    # optimized transpilation of circuit to native gates
    optimized_circuit = custom_pass_manager.run(transpiled_circuit_unoptimized)

    # simulate the optimized circuit
    statevector_optimized = Statevector.from_instruction(optimized_circuit)
    probabilities_optimized = np.abs(statevector_optimized.data) ** 2

    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # print(transpiled_circuit_unoptimized)
    # print(optimized_circuit)

    np.testing.assert_allclose(
        probabilities_unoptimized,
        probabilities_optimized,
        atol=1e-3,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n\n"
            f"Circuit: {qc}\n\n",
        ),
    )

    optimized_dag = circuit_to_dag(optimized_circuit)
    assert optimized_dag.depth() == optimized_depth


@pytest.mark.parametrize(
    "gates",
    [
        # [("CXGate", None, [0, 1]), ("CXGate", None, [0, 2]), ("CXGate", None, [0, 3]), ("CXGate", None, [0, 4])],
        # [ ("CHGate", None, [0, 1])], !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # [("HGate", None, [0]), ("CHGate", None, [0, 1]), ("CHGate", None, [0, 2]), ("CHGate", None, [0, 3]), ("CHGate", None, [0, 4])],
        # [("HGate", None, [0]), ("CHGate", None, [0, 1]), ("HGate", None, [1]), ("CHGate", None, [0, 2]), ("HGate", None, [2]), ("CHGate", None, [0, 3]), ("HGate", None, [3]), ("CHGate", None, [0, 4])],
        # [("XGate", None, [0]), ("CHGate", None, [0, 1]), ("XGate", None, [1]), ("CHGate", None, [0, 2]), ("XGate", None, [2]), ("CHGate", None, [0, 3]), ("XGate", None, [3]), ("CHGate", None, [0, 4])],
        # [("SGate", None, [0]), ("CHGate", None, [0, 1]), ("TGate", None, [1]), ("CHGate", None, [1, 2]), ("SGate", None, [2]), ("CHGate", None, [2, 3]), ("TGate", None, [3]), ("CHGate", None, [3, 4])],
        # [("SGate", None, [0]), ("CHGate", None, [0, 1]), ("TGate", None, [1]), ("CHGate", None, [0, 2]), ("XGate", None, [2]), ("CHGate", None, [0, 3]), ("YGate", None, [3]), ("CHGate", None, [0, 4])],
    ],
    ids=lambda val: f"{val}",
)
def test_ionq_optmizer_plugin_five_qubits(gates, optimized_depth):
    custom_pass_manager_plugin = TrappedIonOptimizerPlugin()
    pass_manager_config = PassManagerConfig()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        pass_manager_config,
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(2)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("simulator", gateset="native")
    transpiled_circuit_unoptimized = transpile(
        qc, backend=backend, optimization_level=3
    )

    # simulate the unoptimized circuit
    statevector_unoptimized = Statevector.from_instruction(
        transpiled_circuit_unoptimized
    )
    probabilities_unoptimized = np.abs(statevector_unoptimized.data) ** 2

    # optimized transpilation of circuit to native gates
    optimized_circuit = custom_pass_manager.run(transpiled_circuit_unoptimized)

    # simulate the optimized circuit
    statevector_optimized = Statevector.from_instruction(optimized_circuit)
    probabilities_optimized = np.abs(statevector_optimized.data) ** 2

    np.testing.assert_allclose(
        probabilities_unoptimized,
        probabilities_optimized,
        atol=1e-3,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )

    optimized_dag = circuit_to_dag(optimized_circuit)
    assert optimized_dag.depth() == optimized_depth
