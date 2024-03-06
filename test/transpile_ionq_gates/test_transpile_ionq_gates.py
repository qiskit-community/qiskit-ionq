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

import numpy as np
import pytest
# import qiskit_ionq

from qiskit import BasicAer, QuantumCircuit,  QuantumRegister, ClassicalRegister, transpile, execute
from qiskit_ionq import ionq_provider
from qiskit_ionq import GPIGate, GPI2Gate, MSGate
from qiskit.circuit.library import HGate, CXGate, RXGate, RYGate, RZGate,  SGate, SdgGate, SXGate, SXdgGate, TGate, TdgGate, UGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, IGate, PhaseGate, GlobalPhaseGate

@pytest.mark.parametrize(
    "ideal_results, gates",
    [
        ([0.5, 0.5], [("HGate", None)]),
        ([1, 0], [("HGate", None), ("HGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("HGate", None), ("HGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("ZGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("XGate", None)]),
        ([1, 0], [("PhaseGate", np.pi)]),
        ([1, 0], [("PhaseGate", np.pi/5)]),
        ([0.5, 0.5], [("HGate", None), ("PhaseGate", np.pi)]),
        ([0.5, 0.5], [("HGate", None), ("PhaseGate", np.pi/7)]),
        ([0, 1], [("XGate", None), ("PhaseGate", np.pi/5)]),
        ([0, 1], [("PhaseGate", np.pi/5), ("XGate", None), ("PhaseGate", np.pi/5)]),
        ([0, 1], [("XGate", None)]),
        ([1, 0], [("XGate", None), ("XGate", None)]),
        ([0, 1], [("XGate", None), ("XGate", None), ("XGate", None)]),
        ([0, 1], [("YGate", None)]),
        ([0, 1], [("YGate", None), ("ZGate", None)]),
        ([1, 0], [("YGate", None), ("YGate", None)]),
        ([0, 1], [("YGate", None), ("YGate", None), ("YGate", None)]),
        ([1, 0], [("ZGate", None)]),
        ([1, 0], [("ZGate", None), ("ZGate", None)]),
        ([1, 0], [("ZGate", None), ("ZGate", None), ("ZGate", None)]),
        ([1, 0], [("XGate", None), ("YGate", None)]),
        ([0, 1], [("XGate", None), ("ZGate", None)]),
        ([1, 0], [("XGate", None), ("YGate", None), ("ZGate", None)]),
        ([1, 0], [("IGate", None)]),
        ([1, 0], [("IGate", None), ("IGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("IGate", None)]),
        ([1, 0], [("HGate", None), ("IGate", None), ("HGate", None)]),
        ([1, 0], [("HGate", None), ("HGate", None)]),
        ([1, 0], [("RXGate", 0)]),
        ([0, 1], [("RXGate", np.pi)]),
        ([0.5, 0.5], [("RXGate", np.pi/2)]),
        ([0, 1], [("RXGate", np.pi/3), ("RXGate", np.pi/3), ("RXGate", np.pi/3)]),
        ([1, 0], [("RYGate", 0)]),
        ([0, 1], [("RYGate", np.pi)]),
        ([0.5, 0.5], [("RYGate", np.pi/2)]),
        ([0, 1], [("RYGate", np.pi/3), ("RYGate", np.pi/3), ("RYGate", np.pi/3)]),




        # ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]))),
        # ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
        # ([1, 0], cirq.Circuit(cirq.S(qubit1[0]))),
        # ([1, 0], cirq.Circuit(cirq.T(qubit1[0]))),
        # ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]))),
        # ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.Y(qubit1[0]))),
        # ([0, 1], cirq.Circuit(cirq.H(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
        # ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.S(qubit1[0]))),
        # ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.T(qubit1[0]))),
        # ([1, 0], cirq.Circuit(cirq.X(qubit1[0]), cirq.X(qubit1[0]))),
        # ([0, 1], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]))),
        # ([0, 1], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
        # ([1, 0], cirq.Circuit(cirq.S(qubit1[0]), cirq.S(qubit1[0]))),
        # ([1, 0], cirq.Circuit(cirq.S(qubit1[0]), cirq.S(qubit1[0]), cirq.S(qubit1[0]))),
        # ([1, 0], cirq.Circuit(cirq.H(qubit1[0]), cirq.H(qubit1[0]), cirq.S(qubit1[0]))),
        # ([1, 0], cirq.Circuit(cirq.H(qubit1[0]), cirq.H(qubit1[0]), cirq.T(qubit1[0]))),
        # ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.H(qubit1[0]), cirq.H(qubit1[0]))),
        # ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]))),
        # ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
    ],
)
def test_transpiling_one_qubit_circuits_to_native_gates(ideal_results, gates):
    # create a quantum circuit
    qr = QuantumRegister(1)
    circuit = QuantumCircuit(qr)
    for gate_name, param in gates:
        gate = eval(gate_name)
        if param is not None:
            circuit.append(gate(param), [0])
        else:
            circuit.append(gate(), [0])

    # transpile circuit to native gates
    provider = ionq_provider.IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
    transpiled_circuit = transpile(circuit, backend)

    # simulate the circuit
    simulator = BasicAer.get_backend('statevector_simulator')
    result = execute(transpiled_circuit, simulator).result()
    statevector = result.get_statevector()
    probabilities = np.abs(statevector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

