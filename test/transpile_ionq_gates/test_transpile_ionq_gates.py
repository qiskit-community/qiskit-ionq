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
import collections.abc

from qiskit import BasicAer, QuantumCircuit,  QuantumRegister, ClassicalRegister, transpile, execute
from qiskit_ionq import ionq_provider
from qiskit.circuit.library import HGate, CXGate, RXGate, RYGate, RZGate,  SGate, SdgGate, SXGate, SXdgGate, TGate, TdgGate, UGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, IGate, PhaseGate


@pytest.mark.parametrize(
    "ideal_results, gates",
    [
        # ([0.707 + 0j, 0.707 + 0j], [("HGate", None)]),
        # ([0 + 0j, 1 + 0j], [("XGate", None)]),
        # ([0.707 + 0j, 0.707 + 0j], [("HGate", None), ("XGate", None)]),
        # ([0 + 0j, 0 + 1j], [("YGate", None)]),
        # ([0 - 0.707j, 0 + 0.707j], [("HGate", None), ("YGate", None)]),
        # ([0 + 1j, 0 + 0j], [("ZGate", None)]),
        # ([0 + 0.707j, 0 - 0.707j], [("HGate", None), ("ZGate", None)]),
        # ([1 + 0j, 0 + 0j], [("IGate", None)]),
        # ([0.707 + 0j, 0.707 + 0j], [("HGate", None), ("IGate", None)]),
        # ([0 + 0.877j, 0.479 + 0j], [("RXGate", 1)]),
        # ([0 + 0.877j, 0 + 0.479j], [("RYGate", 1)]),
        # ([0.877 - 0.479j, 0 + 0j], [("RZGate", 1)]),
        # ([0.707 - 0.707j, 0 + 0j], [("SGate", None)]),
        # ([0.5 - 0.5j, 0.5 + 0.5j], [("HGate", None), ("SGate", None)]),
        # ([0.707 + 0.707j, 0 + 0j], [("SdgGate", None)]),
        # ([0.5 + 0.5j, 0.5 - 0.5j], [("HGate", None), ("SdgGate", None)]),
        # ([0.9238 - 0.3826j, 0 + 0j], [("TGate", None)]),
        # ([0.653 - 0.270j, 0.653 + 0.270j], [("HGate", None), ("TGate", None)]),
        # ([0.9238 + 0.3826j, 0 + 0j], [("TdgGate", None)]),
        # ([0.653 + 0.270j, 0.653 - 0.270j], [("HGate", None), ("TdgGate", None)]),
        # ([0.707 + 0j, 0 - 0.707j], [("SXGate", None)]),
        # ([0.5 - 0.5j, 0.5 - 0.5j], [("HGate", None), ("SXGate", None)]),
        # ([0.707 + 0j, 0 + 0.707j], [("SXdgGate", None)]),
        # ([0.5 + 0.5j, 0.5 + 0.5j], [("HGate", None), ("SXdgGate", None)]),
        #([0.707 + 0j, -0.294 + 0.6429j], [("HGate", None), ("U1Gate", 2)]),
        #([0.707 + 0j, -0.294 + 0.6429j], [("HGate", None), ("PhaseGate", 2)]),
        # add U3, remove rest
    ]
)
def test_statevector_after_transpile(ideal_results, gates):
    # create a quantum circuit
    qr = QuantumRegister(1)
    circuit = QuantumCircuit(qr)
    for gate_name, param in gates:
        gate = eval(gate_name)
        if param is not None:
            if isinstance(param, collections.abc.Sequence):
                circuit.append(gate(*param), [0])
            else:
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
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", statevector)
    np.testing.assert_allclose(statevector, ideal_results, atol=1e-3)


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
        ([0.608559, 0.391441], [("HGate", None), ("RZGate", 0.5), ("SGate", None), ("RXGate", 0.25)]),
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
        ([0.5, 0.5], [("HGate", None), ("U2Gate", [0, 0]), ("HGate", None)]),
        ([0.117426, 0.882574], [("HGate", None), ("U2Gate", [1, 2]), ("HGate", None)]),
        ([0.611645, 0.388255], [("HGate", None), ("U2Gate", [2, 1]), ("HGate", None), ("U1Gate", 2.5), ("HGate", None),]),
        ([0.611645, 0.388255], [("HGate", None), ("U2Gate", [2, 1]), ("HGate", None), ("PhaseGate", 2.5), ("HGate", None),]),
        ([0.547137, 0.452862], [("HGate", None), ("U3Gate", [1, 2, 3]), ("HGate", None)]),
        ([0.770151, 0.229849], [("U3Gate", [1, 2, 3])]),
        ([0.453940, 0.546059], [("HGate", None), ("U3Gate", [1, 2, 3]), ("HGate", None), ("U3Gate", [3, 2, 1])]),
        ([0.180872, 0.819128], [("HGate", None), ("U2Gate", [1, 2]), ("HGate", None), ("U2Gate", [2, 1])]),
        ([0.950819, 0.049181], [("HGate", None), ("U3Gate", [1, 2, 3]), ("HGate", None), ("U2Gate", [2, 3])]),
        ([0.571019, 0.428981], [("HGate", None), ("U2Gate", [0.2, 0.1]), ("HGate", None), ("U1Gate", 1.5), ("HGate", None), ("U3Gate", [0.2, 0.4, 0.6])]),
        ([0.571019, 0.428981], [("HGate", None), ("U2Gate", [0.2, 0.1]), ("HGate", None), ("PhaseGate", 1.5), ("HGate", None), ("U3Gate", [0.2, 0.4, 0.6])]),
        ([0.019810, 0.980190], [("HGate", None), ("U2Gate", [0.2, 0.1]), ("TGate", None), ("U1Gate", 1.5), ("TdgGate", None), ("U3Gate", [0.2, 0.4, 0.6])]),
        ([0.291751, 0.708249], [("HGate", None), ("U2Gate", [1.2, 1.1]), ("TGate", None), ("U1Gate", 0.65), ("TdgGate", None), ("U3Gate", [1.2, 1.4, 1.6])]),
        ([0.424045, 0.575955], [("HGate", None), ("U2Gate", [1.2, 1.1]), ("TGate", None), ("U1Gate", 0.65), ("SXdgGate", None), ("U3Gate", [1.2, 1.4, 1.6])]),
        ([0.778976, 0.221024], [("HGate", None), ("U2Gate", [1.2, 1.1]), ("TdgGate", None), ("U1Gate", 0.65), ("SXGate", None), ("U3Gate", [1.2, 1.4, 1.6])]),
        ([0.158230, 0.841770], [("HGate", None), ("U2Gate", [1.2, 1.1]), ("SGate", None), ("U3Gate", [0.5, 0.6, 0.7]), ("SdgGate", None), ("U3Gate", [1.2, 1.4, 1.6])]),
        ([0.158230, 0.841770], [("HGate", None), ("U2Gate", [1.2, 1.1]), ("IGate", None), ("SGate", None), ("U3Gate", [0.5, 0.6, 0.7]), ("SdgGate", None), ("U3Gate", [1.2, 1.4, 1.6])]),
        ([0.757411, 0.242589], [("HGate", None), ("RXGate", 0.7), ("IGate", None), ("SGate", None), ("RZGate", 1.22), ("SdgGate", None), ("U3Gate", [1.2, 1.4, 1.6]),  ("RYGate", 1.4),]),
        ([1, 0], [("IGate", None)]),
        ([0.883156, 0.116844], [("U3Gate", [1.2, 1.6, 1.8]), ("TGate", None), ("RYGate", 1)]),
        ([0.180237, 0.819763], [("U3Gate", [1.2, 1.6, 1.8]), ("TGate", None), ("HGate", None), ("SXGate", None)]),
    ],
)
def test_transpiling_one_qubit_circuits_to_native_gates(ideal_results, gates):
    # create a quantum circuit
    qr = QuantumRegister(1)
    circuit = QuantumCircuit(qr)
    for gate_name, param in gates:
        gate = eval(gate_name)
        if param is not None:
            if isinstance(param, collections.abc.Sequence):
                circuit.append(gate(*param), [0])
            else:
                circuit.append(gate(param), [0])
        else:
            circuit.append(gate(), [0])

    # transpile circuit to native gates
    provider = ionq_provider.IonQProvider()
    backend = provider.get_backend("ionq_simulator", gateset="native")
    transpiled_circuit = transpile(circuit, backend)
    ## TODO: Remove:
    #print(transpiled_circuit)

    # simulate the circuit
    simulator = BasicAer.get_backend('statevector_simulator')
    result = execute(transpiled_circuit, simulator).result()
    statevector = result.get_statevector()
    probabilities = np.abs(statevector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)
