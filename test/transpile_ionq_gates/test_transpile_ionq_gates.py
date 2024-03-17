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
        ([0.770151, 0.229849], [("HGate", None), ("PhaseGate", 1), ("HGate", None)]),
        ([0.535369, 0.464631], [("HGate", None), ("PhaseGate", 1.5), ("HGate", None)]),
        ([0, 1], [("XGate", None)]),
        ([1, 0], [("XGate", None), ("XGate", None)]),
        ([0, 1], [("XGate", None), ("XGate", None), ("XGate", None)]),
        ([0, 1], [("YGate", None)]),
        ([0, 1], [("YGate", None), ("ZGate", None)]),
        ([1, 0], [("YGate", None), ("YGate", None)]),
        ([0, 1], [("YGate", None), ("YGate", None), ("YGate", None)]),
        ([1, 0], [("YGate", None), ("ZGate", None), ("YGate", None)]),
        ([0, 1], [("XGate", None), ("YGate", None), ("XGate", None)]),
        ([0.848353, 0.151647], [("HGate", None), ("YGate", None), ("PhaseGate", 0.8), ("YGate", None), ("HGate", None)]),
        ([0.848353, 0.151647], [("HGate", None), ("YGate", None), ("U1Gate", 0.8), ("YGate", None), ("HGate", None)]),
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
        ([0.770151, 0.229849], [("HGate", None), ("RYGate", 1), ("HGate", None)]),
        ([1, 0], [("HGate", None), ("RYGate", 1), ("HGate", None), ("RYGate", 1),]),
        ([0.645963, 0.354037], [("HGate", None), ("RYGate", 1), ("HGate", None), ("RXGate", 1),]),
        ([0.454676, 0.545324], [("HGate", None), ("RYGate", 1), ("RXGate", 1), ("RYGate", 1), ("HGate", None)]),
        ([1, 0], [("RZGate", 0)]),
        ([1, 0], [("RZGate", np.pi)]),
        ([0, 1], [("XGate", None), ("RZGate", np.pi/2)]),
        ([0.770151, 0.229849], [("HGate", None), ("RZGate", 1), ("HGate", None)]),
        ([0.229849, 0.770151], [("HGate", None), ("RZGate", 1), ("YGate", None), ("HGate", None)]),
        ([1, 0], [("YGate", None), ("RZGate", 1), ("YGate", None)]),
        ([0.614924, 0.385076], [("HGate", None), ("RZGate", 1/2), ("HGate", None), ("RZGate", 1/2), ("HGate", None)]),
        ([1, 0], [("SGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("SGate", None)]),
        ([0.885076, 0.114924], [("RXGate", 0.5), ("SGate", None), ("RXGate", 0.5)]),
        ([0.885076, 0.114924], [("RXGate", 0.5), ("SGate", None), ("RXGate", 0.5)]),
        ([0.289632, 0.710368], [("HGate", None), ("RYGate", 0.5), ("SGate", None), ("RYGate", 0.5)]),
        ([0.267739, 0.732261], [("HGate", None), ("RYGate", 0.5), ("SGate", None), ("RYGate", 0.25)]),
        ([0.376298, 0.623702], [("HGate", None), ("RYGate", 0.5), ("SGate", None), ("RXGate", 0.25)]),
        ([0.739713, 0.260287], [("HGate", None), ("RXGate", 0.5), ("SGate", None), ("RXGate", 0.5)]),
        ([0.770151, 0.229849], [("RXGate", 1)]),
        ([0.770151, 0.229849], [("RYGate", 1)]),
        ([1, 0], [("RZGate", 1)]),
        ([0.079264, 0.920735], [("HGate", None), ("RYGate", 1), ("SGate", None)]),
        ([0.272676, 0.727324], [("HGate", None), ("RYGate", 1), ("SGate", None), ("RYGate", 1)]),
        ([0.5, 0.5], [("HGate", None), ("RYGate", 1), ("SGate", None), ("RXGate", 1)]),
        ([0.954649, 0.045351], [("RXGate", 2), ("SGate", None), ("HGate", None)]),
        ([0.045351, 0.954649], [("RXGate", 2), ("SdgGate", None), ("HGate", None)]),
        ([0.5, 0.5], [("RXGate", 2), ("SGate", None), ("SdgGate", None), ("HGate", None)]),
        ([0.5, 0.5], [("RXGate", 2), ("SGate", None), ("SdgGate", None), ("HGate", None)]),
        ([0.826145, 0.173855], [("RXGate", 2), ("SGate", None), ("RZGate", 0.8), ("SdgGate", None), ("HGate", None)]),
        ([0.260287, 0.739713], [("HGate", None), ("SXGate", None), ("RYGate", 0.5)]),
        ([0.5, 0.5], [("HGate", None), ("SXGate", None), ("RYGate", 0.5), ("SXdgGate", None)]),
        ([0.938791, 0.061209], [("HGate", None), ("SXGate", None), ("RYGate", 0.5), ("HGate", None)]),
        ([0.810272, 0.189728], [("HGate", None), ("TGate", None), ("RYGate", 0.5), ("HGate", None)]),
        ([0.674355, 0.325645], [("HGate", None), ("TdgGate", None), ("RZGate", 2), ("HGate", None)]),
        ([0.5, 0.5], [("SXGate", None)]),
        ([1, 0], [("SGate", None)]),
        ([1, 0], [("TGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("SXGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("YGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("SGate", None)]),
        ([0.5, 0.5], [("HGate", None), ("TGate", None)]),
        ([1, 0], [("XGate", None), ("XGate", None)]),
        ([0, 1], [("SXGate", None), ("SXGate", None)]),
        ([0, 1], [("SXdgGate", None), ("SXdgGate", None)]),
        ([1, 0], [("SXdgGate", None), ("SXGate", None)]),
        ([0.5, 0.5], [("SXdgGate", None), ("SXdgGate", None), ("SXdgGate", None)]),
        ([1, 0], [("SGate", None), ("SGate", None)]),
        ([1, 0], [("SGate", None), ("SGate", None), ("SGate", None)]),
        ([1, 0], [("HGate", None), ("HGate", None), ("SGate", None)]),
        ([1, 0], [("HGate", None), ("HGate", None), ("TGate", None)]),
        ([0, 1], [("YGate", None), ("SGate", None), ("SGate", None)]),
        ([1, 0], [("TGate", None)]),
        ([1, 0], [("TdgGate", None)]),
        ([1, 0], [("TGate", None), ("TGate", None)]),
        ([1, 0], [("TdgGate", None), ("TdgGate", None)]),
        ([1, 0], [("TGate", None), ("TGate", None), ("TGate", None)]),
        ([1, 0], [("TdgGate", None), ("TdgGate", None), ("TdgGate", None)]),
        ([0, 1], [("YGate", None), ("TdgGate", None), ("TdgGate", None)]),
        ([0.604379, 0.395622], [("YGate", None), ("RXGate", 1), ("TGate", None), ("RXGate", 1)]),
        ([0.604379, 0.395622], [("YGate", None), ("RXGate", 1), ("TdgGate", None), ("RXGate", 1)]),
        ([0.410401, 0.589599], [("YGate", None), ("RXGate", 1), ("HGate", None), ("TdgGate", None), ("RXGate", 1)]),
        ([1, 0], [("U1Gate", np.pi)]),
        ([1, 0], [("U1Gate", np.pi/5)]),
        ([0.5, 0.5], [("HGate", None), ("U1Gate", np.pi)]),
        ([0.5, 0.5], [("HGate", None), ("U1Gate", np.pi/7)]),
        ([0, 1], [("XGate", None), ("U1Gate", np.pi/5)]),
        ([0, 1], [("U1Gate", np.pi/5), ("XGate", None), ("U1Gate", np.pi/5)]),
        ([0.770151, 0.229849], [("HGate", None), ("U1Gate", 1), ("HGate", None)]),
        ([0.535369, 0.464631], [("HGate", None), ("U1Gate", 1.5), ("HGate", None)]),

        
        #TODO: test T  and Tdag



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
    #np.testing.assert_allclose(99, statevector)
    probabilities = np.abs(statevector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

