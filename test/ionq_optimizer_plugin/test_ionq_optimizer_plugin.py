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
    transpile,
)
from qiskit.converters import circuit_to_dag
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
from qiskit_ionq import GPIGate, GPI2Gate, MSGate, ZZGate
from qiskit_ionq import (
    IonQProvider,
    TrappedIonOptimizerPlugin,
    TrappedIonOptimizerPluginSimpleRules,
    TrappedIonOptimizerPluginCompactGates,
    TrappedIonOptimizerPluginCommuteGpi2ThroughMs,
)

# Mapping from gate names to gate classes
gate_map = {
    # IonQ gates
    "GPIGate": GPIGate,
    "GPI2Gate": GPI2Gate,
    "MSGate": MSGate,
    "ZZGate": ZZGate,
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
    if gate_name == "measure":
        for qubit in qubits:
            circuit.measure(qubit, qubit)
    else:
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
        # GPI2(phi) * GPI2(phi + 0.5) on different qubits do not cancel
        ([("GPI2Gate", [1], [0]), ("GPI2Gate", [1.5], [2])], 1),
        # GPI2(phi) * MS *  GPI2(phi + 0.5) do not cancel
        (
            [
                ("GPI2Gate", [1], [0]),
                ("MSGate", [0.2, 0.3, 0.25], [0, 2]),
                ("GPI2Gate", [1.5], [0]),
            ],
            3,
        ),
        # GPI2(phi) * MS *  GPI2(phi + 0.5) do cancel if MS
        # applies to qubits 0 and 2 while GPIs apply to qubit 1
        (
            [
                ("GPI2Gate", [1], [1]),
                ("MSGate", [0.2, 0.3, 0.25], [0, 2]),
                ("GPI2Gate", [1.5], [1]),
            ],
            1,
        ),
        # GPI2(phi) * GPI2(phi + 0.5) but on different qubits do not cancel
        ([("GPI2Gate", [1], [1]), ("GPI2Gate", [1.5], [2])], 1),
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
        # GPI(phi) * GPI(phi) do not cancel if applied to different qubits
        ([("GPIGate", [1], [0]), ("GPIGate", [1], [1])], 1),
        # GPI(phi) * GPI(phi) do not cancel if MS between them
        (
            [
                ("GPIGate", [1], [0]),
                ("MSGate", [0.2, 0.3, 0.25], [0, 1]),
                ("GPIGate", [1], [0]),
            ],
            3,
        ),
        # GPI(phi) * GPI(phi) do  cancel if MS between them applies to different qubits
        (
            [
                ("GPIGate", [1], [1]),
                ("MSGate", [0.2, 0.3, 0.25], [0, 2]),
                ("GPIGate", [1], [1]),
            ],
            1,
        ),
        # GPI(phi) * GPI(phi) on different qubits do not cancel
        ([("GPIGate", [1], [0]), ("GPIGate", [1], [1])], 1),
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
        # Four GPI2 with equal arguments but on different qubits do not cancel
        (
            [
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [0.7], [2]),
                ("GPI2Gate", [0.7], [0]),
            ],
            2,
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
        # Two GPI2 with equal arguments equal GPI on different qubits stay unchanged
        ([("GPI2Gate", [1.7], [0]), ("GPI2Gate", [1.7], [1])], 1),
        # Two GPI2 with equal arguments equal GPI but with MS between stay unchanged
        (
            [
                ("GPI2Gate", [1.7], [0]),
                ("MSGate", [0.2, 0.3, 0.25], [0, 1]),
                ("GPI2Gate", [1.7], [0]),
            ],
            3,
        ),
        # Two GPI2 with equal arguments equal GPI but with MS
        # between an different qubits will be replaced by GPI
        (
            [
                ("GPI2Gate", [1.7], [0]),
                ("MSGate", [0.2, 0.3, 0.25], [2, 1]),
                ("GPI2Gate", [1.7], [0]),
            ],
            1,
        ),
        # Two GPI2 * GPI2 * GPI = GPI * GPI = Id
        (
            [
                ("GPI2Gate", [1.7], [1]),
                ("GPI2Gate", [1.7], [1]),
                ("GPIGate", [1.7], [1]),
            ],
            0,
        ),
        # Two GPI2 * GPI2 * GPI  not all an the same qubit stay the same
        (
            [
                ("GPI2Gate", [1.7], [1]),
                ("GPI2Gate", [1.7], [2]),
                ("GPIGate", [1.7], [1]),
            ],
            2,
        ),
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
        # Two GPI2 with equal arguments but different qubits are not merged
        (
            [
                ("GPIGate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [1]),
            ],
            2,
        ),
        (
            [
                ("GPIGate", [1.7], [1]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [1]),
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
        # GPI * GPI2 * GPI2 * (GPI2 * GPI2-dag)
        # -> GPI * GPI * (GPI2 * GPI2-dag)
        # -> * (GPI2 * GPI2-dag) -> Id
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
        # same as above but not all on the same qubit
        (
            [
                ("GPIGate", [1.7], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.7], [2]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [1.2], [1]),
            ],
            1,
        ),
        # GPI2-dag * GPI * GPI2 * GPI2 * GPI2
        # -> GPI2-dag * GPI * GPI * GPI2
        # -> GPI2-dag * GPI2 -> Id
        (
            [
                ("GPI2Gate", [2.2], [1]),
                ("GPIGate", [1.7], [1]),
                ("GPI2Gate", [1.7], [1]),
                ("GPI2Gate", [1.7], [1]),
                ("GPI2Gate", [1.7], [1]),
            ],
            0,
        ),
    ],
    ids=lambda val: f"{val}",
)
def test_ionq_optimizer_plugin_simple_one_qubit_rules(gates, optimized_depth):  # pylint: disable=invalid-name
    """Test TrappedIonOptimizerPluginSimpleRules."""

    ############################################################
    # First test TrappedIonOptimizerPluginSimpleRules
    # to test the following transformation passes in isolation:
    #    - CancelGPI2Adjoint
    #    - CancelGPIAdjoint
    #    - GPI2TwiceIsGPI
    #############################################################

    custom_pass_manager_plugin = TrappedIonOptimizerPluginSimpleRules()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(3)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )

    optimized_dag = circuit_to_dag(optimized_circuit)
    assert optimized_dag.depth() == optimized_depth

    ###################################################
    # Second, test TrappedIonOptimizerPlug
    ###################################################

    custom_pass_manager_plugin = TrappedIonOptimizerPlugin()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(3)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )


@pytest.mark.parametrize(
    "gates, optimized_depth",
    [
        # any combination of GPI/GPI2 gates on the
        # same qubit is collaped to 3 gates
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
                ("GPI2Gate", [1.8], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [1.1], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [2.7], [0]),
            ],
            3,
        ),
        # any combination of GPI/GPI2 gates on the
        # same qubit is collaped to 3 gates
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
        # any combination of GPI/GPI2 gates on the
        # same qubit is collaped to 3 gates
        (
            [
                ("GPI2Gate", [1.8], [0]),
                ("GPI2Gate", [1.7], [0]),
                ("GPI2Gate", [0.1], [0]),
                ("GPI2Gate", [2.7], [0]),
                ("GPI2Gate", [2.7], [0]),
                ("GPI2Gate", [1.7], [0]),
            ],
            3,
        ),
        # any combination of GPI/GPI2 gates on the
        # same qubit is collaped to 3 gates
        (
            [
                ("GPIGate", [1.8], [0]),
                ("GPI2Gate", [1.7], [1]),
                ("GPI2Gate", [0.1], [0]),
                ("GPI2Gate", [2.7], [1]),
                ("GPI2Gate", [2.7], [0]),
                ("GPIGate", [1.7], [0]),
            ],
            3,
        ),
        # any combination of GPI/GPI2 gates on the
        # same qubit is collaped to 3 gates
        (
            [
                ("GPI2Gate", [0.2], [0]),
                ("GPIGate", [0.7], [1]),
                ("GPI2Gate", [1.5], [0]),
                ("GPIGate", [2.8], [1]),
            ],
            2,
        ),
        # any combination of GPI/GPI2 gates on the
        # same qubit is collaped to 3 gates
        (
            [
                ("GPI2Gate", [0.2], [0]),
                ("GPIGate", [0.7], [0]),
                ("GPI2Gate", [1.5], [0]),
                ("GPIGate", [2.8], [0]),
            ],
            3,
        ),
        # combine one qubit gates with two qubit gates
        (
            [
                ("MSGate", [0.2, 0.3, 0.25], [0, 1]),
                ("GPIGate", [0.2], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [1.5], [0]),
                ("GPIGate", [2.8], [0]),
                ("MSGate", [1.2, 1.3, 1.25], [0, 1]),
                ("GPIGate", [0.2], [1]),
                ("GPIGate", [0.7], [1]),
                ("GPI2Gate", [1.5], [1]),
                ("GPI2Gate", [0.8], [1]),
                ("GPI2Gate", [0.7], [1]),
            ],
            8,
        ),
        # same as above but MS apply to qubits 0 and 2
        (
            [
                ("MSGate", [0.2, 0.3, 0.25], [0, 2]),
                ("GPIGate", [0.2], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [1.5], [0]),
                ("GPIGate", [2.8], [0]),
                ("MSGate", [1.2, 1.3, 1.25], [0, 2]),
                ("GPIGate", [0.2], [1]),
                ("GPIGate", [0.7], [1]),
                ("GPI2Gate", [1.5], [1]),
                ("GPI2Gate", [0.8], [1]),
                ("GPI2Gate", [0.7], [1]),
            ],
            5,
        ),
        # simililar to the two tests above bit after second
        # MS gate not all gates are lined on the same qubit
        (
            [
                ("MSGate", [0.2, 0.3, 0.25], [0, 1]),
                ("GPIGate", [0.2], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("GPI2Gate", [1.5], [0]),
                ("GPIGate", [2.8], [0]),
                ("MSGate", [1.2, 1.3, 1.25], [0, 1]),
                ("GPIGate", [0.2], [0]),
                ("GPIGate", [0.7], [0]),
                ("GPI2Gate", [1.5], [1]),
                ("GPI2Gate", [0.8], [1]),
                ("GPI2Gate", [0.75], [2]),
                ("GPI2Gate", [0.7], [2]),
            ],
            7,
        ),
        # all GPI and GPI2 gates are expected
        # to be consolidated as GPI2 * GPI * GPI2
        (
            [
                ("MSGate", [0.2, 0.3, 0.25], [0, 2]),
                ("GPIGate", [3.2], [1]),
                ("GPI2Gate", [0.1], [1]),
                ("GPI2Gate", [1.5], [1]),
                ("GPIGate", [2.4], [1]),
                ("MSGate", [1.2, 1.3, 1.25], [0, 2]),
                ("GPIGate", [1.2], [1]),
                ("GPIGate", [2.7], [1]),
                ("GPI2Gate", [0.5], [1]),
                ("GPI2Gate", [1.8], [1]),
                ("GPI2Gate", [1.75], [1]),
                ("GPI2Gate", [1.7], [1]),
            ],
            3,
        ),
        # the first sequence of 5 GPI and GPI2 is expected
        # and the second sequence of 6 GPI and GPI2 is expected
        # to be consolidated as GPI2 * GPI * GPI2 separately
        (
            [
                ("MSGate", [0.2, 0.3, 0.25], [0, 2]),
                ("GPIGate", [3.2], [1]),
                ("GPI2Gate", [0.1], [1]),
                ("GPI2Gate", [1.5], [1]),
                ("GPIGate", [2.4], [1]),
                ("GPI2Gate", [0.2], [1]),
                ("MSGate", [1.2, 1.3, 1.25], [0, 1]),
                ("GPIGate", [1.2], [1]),
                ("GPIGate", [2.7], [1]),
                ("GPI2Gate", [0.5], [1]),
                ("GPI2Gate", [1.8], [1]),
                ("GPI2Gate", [1.75], [1]),
                ("GPI2Gate", [1.7], [1]),
            ],
            7,
        ),
    ],
    ids=lambda val: f"{val}",
)
def test_ionq_optimizer_plugin_compact_more_than_three_gates(gates, optimized_depth):  # pylint: disable=invalid-name
    """Test TrappedIonOptimizerPluginCompactGates."""

    ###############################################################
    # First test TrappedIonOptimizerPluginCompactGates
    # to test the following transformation passes in isolation:
    #    - CompactMoreThanThreeSingleQubitGates
    ###############################################################

    custom_pass_manager_plugin = TrappedIonOptimizerPluginCompactGates()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(3)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}",
        ),
    )

    optimized_dag = circuit_to_dag(optimized_circuit)
    assert optimized_dag.depth() == optimized_depth

    ###################################################
    # Second, test TrappedIonOptimizerPlugin
    ###################################################

    custom_pass_manager_plugin = TrappedIonOptimizerPlugin()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(3)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}",
        ),
    )


@pytest.mark.parametrize(
    "gates, optimized_depth",
    [
        # testing GPI gates
        (
            [
                ("GPIGate", [0], [0]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("HGate", None, [1]),
                ("GPIGate", [0], [-0]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            5,
        ),
        (
            [
                ("GPIGate", [0], [1]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("HGate", None, [1]),
                ("GPIGate", [0], [1]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            5,
        ),
        (
            [
                ("GPIGate", [0.5], [0]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("HGate", None, [1]),
                ("GPIGate", [0.5], [0]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            5,
        ),
        (
            [
                ("GPIGate", [0.5], [1]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("HGate", None, [1]),
                ("GPIGate", [0.5], [1]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            5,
        ),
        (
            [
                ("GPIGate", [-0.5], [0]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("HGate", None, [1]),
                ("GPIGate", [-0.5], [0]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            5,
        ),
        (
            [
                ("GPIGate", [-0.5], [1]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("HGate", None, [1]),
                ("GPIGate", [-0.5], [1]),
                ("MSGate", [0, 0, 0.25], [0, 1]),
            ],
            5,
        ),
        # testing GPI2 gates
        (
            [
                ("GPI2Gate", [0], [0]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("XGate", None, [2]),
                ("GPI2Gate", [0], [0]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            6,
        ),
        (
            [
                ("GPI2Gate", [0], [2]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("XGate", None, [2]),
                ("GPI2Gate", [0], [2]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            7,
        ),
        (
            [
                ("GPI2Gate", [0.5], [0]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("XGate", None, [2]),
                ("GPI2Gate", [0.5], [0]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            6,
        ),
        (
            [
                ("GPI2Gate", [0.5], [2]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("XGate", None, [2]),
                ("GPI2Gate", [0.5], [2]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            7,
        ),
        (
            [
                ("GPI2Gate", [-0.5], [0]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("XGate", None, [2]),
                ("GPI2Gate", [-0.5], [0]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            6,
        ),
        (
            [
                ("GPI2Gate", [-0.5], [2]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            2,
        ),
        (
            [
                ("HGate", None, [0]),
                ("XGate", None, [2]),
                ("GPI2Gate", [-0.5], [2]),
                ("MSGate", [0, 0, 0.25], [0, 2]),
            ],
            7,
        ),
    ],
    ids=lambda val: f"{val}",
)
def test_commute_gpis_through_ms(gates, optimized_depth):
    """Test TrappedIonOptimizerPluginCommuteGpi2ThroughMs."""

    ###########################################################
    # First test TrappedIonOptimizerPluginCommuteGpi2ThroughMs
    # to test the following transformation passes in isolation:
    #    - CommuteGPIsThroughMS
    ###########################################################

    custom_pass_manager_plugin = TrappedIonOptimizerPluginCommuteGpi2ThroughMs()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(3)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )

    optimized_dag = circuit_to_dag(optimized_circuit)
    assert optimized_dag.depth() == optimized_depth
    assert optimized_circuit != transpiled_circuit_unoptimized

    ##############################################
    # Second, test TrappedIonOptimizerPlugin
    ##############################################

    custom_pass_manager_plugin = TrappedIonOptimizerPlugin()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(3)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )


@pytest.mark.parametrize(
    "gates, should_commute",
    [
        (
            [
                ("GPIGate", [0.5], [0]),
                ("GPIGate", [0.7], [0]),
                ("ZZGate", [0.25], [0, 1]),
            ],
            True,
        ),
        (
            [
                ("GPIGate", [0.5], [1]),
                ("GPIGate", [0.7], [1]),
                ("ZZGate", [0.25], [0, 1]),
            ],
            True,
        ),
        (
            [
                ("GPIGate", [0.5], [0]),
                ("GPIGate", [0.7], [1]),
                ("ZZGate", [0.25], [0, 1]),
            ],
            False,
        ),
        (
            [
                ("GPIGate", [0.5], [1]),
                ("GPIGate", [0.7], [0]),
                ("ZZGate", [0.25], [0, 1]),
            ],
            False,
        ),
        (
            [
                ("GPIGate", [0.5], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("ZZGate", [0.25], [0, 1]),
            ],
            False,
        ),
        (
            [
                ("GPI2Gate", [0.5], [0]),
                ("GPIGate", [0.7], [0]),
                ("ZZGate", [0.25], [0, 1]),
            ],
            False,
        ),
        (
            [
                ("GPI2Gate", [0.5], [0]),
                ("GPI2Gate", [0.7], [0]),
                ("ZZGate", [0.25], [0, 1]),
            ],
            False,
        ),
    ],
    ids=lambda val: f"{val}",
)
def test_commute_two_gpi_through_zz(gates, should_commute):
    """Test TrappedIonOptimizerPluginCommuteGpi2ThroughMs."""

    custom_pass_manager_plugin = TrappedIonOptimizerPlugin()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(2)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    provider = IonQProvider()
    backend = provider.get_backend(
        "ionq_simulator", gateset="native", noise_model="forte-1"
    )
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )

    if should_commute:
        assert optimized_circuit != transpiled_circuit_unoptimized
    else:
        assert optimized_circuit == transpiled_circuit_unoptimized


@pytest.mark.parametrize(
    "gates",
    [
        [
            ("HGate", None, [0]),
            ("HGate", None, [1]),
            ("HGate", None, [2]),
            ("CHGate", None, [0, 2]),
            ("XGate", None, [0]),
            ("XGate", None, [1]),
            ("XGate", None, [2]),
            ("HGate", None, [0]),
            ("SGate", None, [1]),
            ("HGate", None, [2]),
            ("CXGate", None, [0, 1]),
        ],
        [
            ("SGate", None, [0]),
            ("HGate", None, [1]),
            ("TGate", None, [2]),
            ("CSGate", None, [0, 2]),
            ("SwapGate", None, [0, 1]),
            ("CHGate", None, [1, 2]),
            ("TGate", None, [0]),
            ("XGate", None, [1]),
            ("SGate", None, [2]),
            ("HGate", None, [0]),
            ("SGate", None, [1]),
            ("XGate", None, [2]),
            ("CU3Gate", [1, 2, 3], [0, 1]),
            ("HGate", None, [0]),
            ("HGate", None, [1]),
            ("HGate", None, [2]),
            ("SwapGate", None, [2, 1]),
        ],
        [
            ("TGate", None, [0]),
            ("XGate", None, [1]),
            ("SwapGate", None, [0, 1]),
            ("HGate", None, [1]),
            ("XGate", None, [2]),
            ("CSGate", None, [0, 2]),
            ("CHGate", None, [1, 2]),
            ("TGate", None, [0]),
            ("XGate", None, [1]),
            ("SwapGate", None, [2, 1]),
            ("SGate", None, [2]),
            ("HGate", None, [0]),
            ("SGate", None, [1]),
            ("XGate", None, [2]),
            ("HGate", None, [0]),
            ("HGate", None, [1]),
            ("HGate", None, [2]),
            ("SwapGate", None, [0, 2]),
            ("TGate", None, [0]),
            ("TGate", None, [1]),
            ("TGate", None, [2]),
            ("SwapGate", None, [0, 1]),
            ("SwapGate", None, [1, 2]),
            ("SwapGate", None, [0, 2]),
        ],
        [
            ("HGate", None, [0]),
            ("XGate", None, [1]),
            ("CHGate", None, [1, 0]),
            ("HGate", None, [1]),
            ("XGate", None, [2]),
            ("CXGate", None, [2, 0]),
            ("CHGate", None, [1, 2]),
            ("HGate", None, [0]),
            ("XGate", None, [1]),
            ("CHGate", None, [2, 1]),
            ("YGate", None, [2]),
            ("HGate", None, [0]),
            ("XGate", None, [1]),
            ("XGate", None, [2]),
            ("CYGate", None, [0, 1]),
            ("HGate", None, [0]),
            ("XGate", None, [1]),
            ("HGate", None, [2]),
            ("CXGate", None, [0, 2]),
            ("XGate", None, [0]),
            ("HGate", None, [1]),
            ("YGate", None, [2]),
            ("CXGate", None, [0, 1]),
            ("CXGate", None, [1, 2]),
            ("CXGate", None, [0, 2]),
        ],
    ],
    ids=lambda val: f"{val}",
)
def test_all_rewrite_rules(gates):
    """Test TrappedIonOptimizerPlugin."""

    custom_pass_manager_plugin = TrappedIonOptimizerPlugin()
    custom_pass_manager = custom_pass_manager_plugin.pass_manager(
        optimization_level=3,
    )

    # create a quantum circuit
    qc = QuantumCircuit(3)
    for gate_name, param, qubits in gates:
        append_gate(qc, gate_name, param, qubits)

    ######################################
    # Testing Aria type devices which
    # transpile native gates using MS gate
    ######################################

    provider = IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
    transpiled_circuit_unoptimized = transpile(
        qc, backend=backend, optimization_level=1
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )

    assert optimized_circuit != transpiled_circuit_unoptimized

    ######################################
    # Testing Forte type devices which
    # transpile native gates using ZZ gate
    ######################################

    provider = IonQProvider()
    backend = provider.get_backend(
        "ionq_simulator", gateset="native", noise_model="forte-1"
    )
    transpiled_circuit_unoptimized = transpile(
        qc, backend=backend, optimization_level=1
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
        atol=1e-5,
        err_msg=(
            f"Unoptmized: {np.round(probabilities_unoptimized, 3)},\n"
            f"Optimized: {np.round(probabilities_optimized, 3)},\n"
            f"Circuit: {qc}"
        ),
    )

    assert optimized_circuit != transpiled_circuit_unoptimized
