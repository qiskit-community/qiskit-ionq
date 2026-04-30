from __future__ import annotations

import math

import pytest
from qiskit.circuit import QuantumCircuit

from qiskit_ionq._translate import translate_native_gates, translate_qis_gates
from qiskit_ionq.gates import GPIGate, MSGate, ZZGate


class TestTranslateToQis:
    def test_h_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        g = translate_qis_gates(qc)
        assert len(g) == 1 and g[0].gate == "h" and g[0].target == 0

    def test_cx_gate(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        g = translate_qis_gates(qc)
        assert g[0].gate == "cnot" and g[0].control == 0 and g[0].target == 1

    def test_rx_gate_with_rotation(self):
        qc = QuantumCircuit(1)
        qc.rx(math.pi / 4, 0)
        g = translate_qis_gates(qc)
        assert g[0].gate == "rx" and math.isclose(g[0].rotation, math.pi / 4)

    def test_rxx_gate(self):
        qc = QuantumCircuit(2)
        qc.rxx(0.5, 0, 1)
        g = translate_qis_gates(qc)
        assert g[0].gate == "xx" and g[0].targets == [0, 1] and math.isclose(g[0].rotation, 0.5)

    def test_measurement_and_barrier_skipped(self):
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.barrier()
        qc.x(0)
        qc.measure(0, 0)
        g = translate_qis_gates(qc)
        assert len(g) == 2 and g[0].gate == "h" and g[1].gate == "x"

    def test_multi_gate_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(math.pi, 1)
        g = translate_qis_gates(qc)
        assert [x.gate for x in g] == ["h", "cnot", "rz"]

    def test_unknown_gate_raises(self):
        qc = QuantumCircuit(1)
        qc.id(0)
        with pytest.raises(ValueError, match="Unsupported"):
            translate_qis_gates(qc)

    @pytest.mark.parametrize(("method", "expected"), [("sx", "v"), ("sdg", "si")])
    def test_gate_name_mapping(self, method, expected):
        qc = QuantumCircuit(1)
        getattr(qc, method)(0)
        assert translate_qis_gates(qc)[0].gate == expected


class TestTranslateToNative:
    def test_gpi_gate(self):
        qc = QuantumCircuit(1)
        qc.append(GPIGate(0.125), [0])
        g = translate_native_gates(qc)
        assert len(g) == 1 and g[0].gate == "gpi" and g[0].target == 0
        assert math.isclose(g[0].phase, 0.125)

    def test_ms_gate(self):
        qc = QuantumCircuit(2)
        qc.append(MSGate(0, 0.5), [0, 1])
        g = translate_native_gates(qc)
        assert g[0].gate == "ms" and g[0].targets == [0, 1]
        assert math.isclose(g[0].phases[0], 0.0) and math.isclose(g[0].phases[1], 0.5)

    def test_ms_angle(self):
        qc = QuantumCircuit(2)
        qc.append(MSGate(0.1, 0.2, 0.15), [0, 1])
        assert math.isclose(translate_native_gates(qc)[0].angle, 0.15)
        qc2 = QuantumCircuit(2)
        qc2.append(MSGate(0, 0), [0, 1])
        assert math.isclose(translate_native_gates(qc2)[0].angle, 0.25)

    def test_zz_gate(self):
        qc = QuantumCircuit(2)
        qc.append(ZZGate(0.3), [0, 1])
        g = translate_native_gates(qc)
        assert g[0].gate == "zz" and g[0].targets == [0, 1] and math.isclose(g[0].phase, 0.3)

    def test_measurement_skipped(self):
        qc = QuantumCircuit(1, 1)
        qc.append(GPIGate(0.5), [0])
        qc.measure(0, 0)
        g = translate_native_gates(qc)
        assert len(g) == 1 and g[0].gate == "gpi"

    def test_unsupported_gate_raises(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        with pytest.raises(ValueError, match="Unsupported"):
            translate_native_gates(qc)
