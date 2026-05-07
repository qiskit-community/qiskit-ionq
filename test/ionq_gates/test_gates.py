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

"""Tests for the IonQ's GPIGate, GPI2Gate, MSGate, ZZGate."""
# pylint: disable=redefined-outer-name

import numpy as np

import pytest

from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, YGate, RXGate, RYGate, HGate
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.quantum_info import Operator
from qiskit_ionq import GPIGate, GPI2Gate, MSGate, ZZGate


@pytest.mark.parametrize("gate,phase", [(XGate(), 0), (YGate(), 0.25)])
def test_gpi_equivalences(gate, phase):
    """Tests equivalence of the GPI gate at specific phases."""
    gpi = GPIGate(phase)
    np.testing.assert_array_almost_equal(gate.to_matrix(), gpi.to_matrix())


@pytest.mark.parametrize(
    "gate,phase", [(RXGate(np.pi / 2), 1), (RYGate(np.pi / 2), 0.25)]
)
def test_gpi2_equivalences(gate, phase):
    """Tests equivalence of the GPI2 gate at specific phases."""
    gpi2 = GPI2Gate(phase)
    np.testing.assert_array_almost_equal(gate.to_matrix(), gpi2.to_matrix())


@pytest.mark.parametrize("gpi2_angle_1, gpi_angle, gpi2_angle_2", [(0, -0.125, 0.5)])
def test_hadamard_equivalence(gpi2_angle_1, gpi_angle, gpi2_angle_2):
    """Tests equivalence of the Hadamard gate with the GPI and GPI2 gates."""
    gpi2_1 = GPI2Gate(gpi2_angle_1)
    gpi = GPIGate(gpi_angle)
    gpi2_2 = GPI2Gate(gpi2_angle_2)
    native_hadamard = np.dot(gpi2_2, np.dot(gpi, gpi2_1))
    np.testing.assert_array_almost_equal(native_hadamard, HGate().to_matrix())


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, np.pi / 2, np.pi, 2 * np.pi])
def test_gpi_inverse(phase):
    """Tests that the GPI gate is unitary."""
    gate = GPIGate(phase)
    mat = np.array(gate)
    np.testing.assert_array_almost_equal(mat.dot(mat.conj().T), np.identity(2))


@pytest.mark.parametrize("phase", [-0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_gpi_definition(phase):
    """Tests equivalence of the GPI gate matrix and its decomposition."""
    gate = GPIGate(phase)
    direct_mat = np.array(gate)
    decomp_mat = Operator(gate.definition).data
    np.testing.assert_array_almost_equal(direct_mat, decomp_mat)


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, np.pi / 2, np.pi, 2 * np.pi])
def test_gpi2_inverse(phase):
    """Tests that the GPI2 gate is unitary."""
    gate = GPI2Gate(phase)

    mat = np.array(gate)
    np.testing.assert_array_almost_equal(mat.dot(mat.conj().T), np.identity(2))


@pytest.mark.parametrize("phase", [-0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_gpi2_definition(phase):
    """Tests equivalence of the GPI2 gate matrix and its decomposition."""
    gate = GPI2Gate(phase)
    direct_mat = np.array(gate)
    decomp_mat = Operator(gate.definition).data
    np.testing.assert_array_almost_equal(direct_mat, decomp_mat)


@pytest.mark.parametrize(
    "params",
    [
        (0, 1, 0.25),
        (0.1, 1, 0.25),
        (0.4, 1, 0.25),
        (np.pi / 2, 0, 0.25),
        (0, np.pi, 0.25),
        (0.1, 2 * np.pi, 0.25),
    ],
)
def test_ms_inverse(params):
    """Tests that the MS gate is unitary."""
    gate = MSGate(params[0], params[1], params[2])

    mat = np.array(gate)
    np.testing.assert_array_almost_equal(mat.dot(mat.conj().T), np.identity(4))


@pytest.mark.parametrize(
    "params",
    [(0, 0, 0.25), (0.1, 0.2, 0.25), (0.5, 0.5, 0.125)],
)
def test_ms_definition(params):
    """Tests equivalence of the MS gate matrix and its decomposition."""
    gate = MSGate(params[0], params[1], params[2])
    direct_mat = np.array(gate)
    decomp_mat = Operator(gate.definition).data
    np.testing.assert_array_almost_equal(direct_mat, decomp_mat)


@pytest.mark.parametrize(
    "angle",
    [0, 0.1, 0.4, np.pi / 2, np.pi, 2 * np.pi],
)
def test_zz_inverse(angle):
    """Tests that the ZZ gate is unitary."""
    gate = ZZGate(angle)

    mat = np.array(gate)
    np.testing.assert_array_almost_equal(mat.dot(mat.conj().T), np.identity(4))


@pytest.mark.parametrize("angle", [-0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_zz_definition(angle):
    """Tests equivalence of the ZZ gate matrix and its decomposition."""
    gate = ZZGate(angle)
    direct_mat = np.array(gate)
    decomp_mat = Operator(gate.definition).data
    np.testing.assert_array_almost_equal(direct_mat, decomp_mat)


def test_qasm3_export():
    """Tests that circuits with IonQ native gates can be exported and parsed as QASM3."""
    import openqasm3.parser

    circuit = QuantumCircuit(2)
    circuit.append(GPIGate(0.25), [0])
    circuit.append(GPI2Gate(0.5), [0])
    circuit.append(MSGate(0.1, 0.2), [0, 1])
    circuit.append(ZZGate(0.3), [0, 1])

    # Should not raise QASM3ExporterError
    qasm3_str = qasm3_dumps(circuit)

    # Verify the output contains our gate definitions
    assert "gate gpi" in qasm3_str
    assert "gate gpi2" in qasm3_str
    assert "gate ms" in qasm3_str
    assert "gate zz" in qasm3_str

    # Should not raise QASM3ParserError (issue #217).
    openqasm3.parser.parse(qasm3_str)


def test_qasm_export_from_transpiled_circuit():
    """Tests QASM export and parse after transpiling to native gateset."""
    import openqasm3.parser
    import qiskit
    import qiskit.qasm2
    import qiskit.qasm3
    from qiskit_ionq import IonQProvider

    # Repro case from issue #217.
    circuit = qiskit.QuantumCircuit(1)
    circuit.u(0.1, 0.2, 0.3, 0)

    provider = IonQProvider()
    backend = provider.get_backend("simulator", gateset="native")

    transpiled = qiskit.transpile(circuit, backend=backend, optimization_level=1)

    # QASM2 export should work.
    qasm2_str = qiskit.qasm2.dumps(transpiled)
    assert "gate gpi(" in qasm2_str or "gate gpi2(" in qasm2_str
    # QASM3 export and parse should work (issue #218).
    qasm3_str = qiskit.qasm3.dumps(transpiled)
    assert "gate gpi(" in qasm3_str or "gate gpi2(" in qasm3_str
    openqasm3.parser.parse(qasm3_str)  # Should not raise QASM3ParserError
