from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from qiskit_ionq.gates import GPI2Gate, GPIGate, MSGate, ZZGate


class TestGPIGate:
    def test_matrix_at_zero(self):
        assert Operator(np.array([[0, 1], [1, 0]])).equiv(Operator(GPIGate(0)))

    def test_matrix_at_quarter(self):
        assert Operator(np.array([[0, -1j], [1j, 0]])).equiv(Operator(GPIGate(0.25)))

    def test_involution(self):
        qc = QuantumCircuit(1)
        qc.append(GPIGate(0.3), [0])
        qc.append(GPIGate(0.3), [0])
        assert Operator(qc).equiv(Operator(np.eye(2)))

    def test_definition_and_unitary(self):
        gate = GPIGate(0.3)
        assert gate.definition is not None and len(gate.definition) > 0
        m = gate.to_matrix()
        np.testing.assert_array_almost_equal(m @ m.conj().T, np.eye(2))

    def test_label(self):
        assert GPIGate(0, label="test").label == "test"


class TestGPI2Gate:
    def test_matrix_at_zero(self):
        s = 1 / math.sqrt(2)
        assert Operator(np.array([[s, -1j * s], [-1j * s, s]])).equiv(Operator(GPI2Gate(0)))

    def test_two_gpi2_equals_gpi(self):
        phi = 0.3
        qc_double = QuantumCircuit(1)
        qc_double.append(GPI2Gate(phi), [0])
        qc_double.append(GPI2Gate(phi), [0])
        qc_single = QuantumCircuit(1)
        qc_single.append(GPIGate(phi), [0])
        assert Operator(qc_double).equiv(Operator(qc_single))

    def test_definition_exists(self):
        gate = GPI2Gate(0.3)
        assert gate.definition is not None and len(gate.definition) > 0

    def test_label(self):
        assert GPI2Gate(0, label="test").label == "test"


class TestMSGate:
    def test_default_is_maximally_entangling(self):
        qc = QuantumCircuit(2)
        qc.rxx(math.pi / 2, 0, 1)
        assert Operator(qc).equiv(Operator(MSGate(0, 0)))

    def test_unitary(self):
        for gate in [MSGate(0.1, 0.3), MSGate(0.1, 0.2)]:
            m = gate.to_matrix()
            np.testing.assert_array_almost_equal(m @ m.conj().T, np.eye(4))

    def test_definition_exists(self):
        gate = MSGate(0, 0)
        assert gate.definition is not None and len(gate.definition) > 0

    def test_params(self):
        g = MSGate(0, 0)
        assert len(g.params) == 3 and g.params[2] == 0.25
        assert MSGate(0.1, 0.2, 0.1).params[2] == 0.1

    def test_label(self):
        assert MSGate(0, 0, label="test").label == "test"


class TestZZGate:
    def test_basics(self):
        gate = ZZGate(0.1)
        assert gate.name == "zz" and gate.num_qubits == 2
        assert gate.definition is not None

    def test_zero_is_identity(self):
        np.testing.assert_array_almost_equal(ZZGate(0).to_matrix(), np.eye(4))

    def test_unitary(self):
        m = ZZGate(0.3).to_matrix()
        np.testing.assert_array_almost_equal(m @ m.conj().T, np.eye(4))

    def test_label(self):
        assert ZZGate(0.1, label="test").label == "test"


@pytest.mark.parametrize(
    "gate",
    [GPIGate(0.3), GPI2Gate(0.15), MSGate(0.1, 0.2, 0.25), ZZGate(0.3)],
    ids=["gpi", "gpi2", "ms", "zz"],
)
def test_definition_matches_array(gate):
    assert Operator(gate).equiv(Operator(gate.definition))
