import json
import unittest

from qiskit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit.providers.ionq import IonQ
from qiskit.providers.ionq.qobj_to_ionq import (build_circuit,
                                                build_output_map, qobj_to_ionq)

from .base import MockCredentialsTestCase


class TestCircuitBuilder(MockCredentialsTestCase):
    def test_measurement_only_circuit(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        qobj = assemble(qc, backend)
        expected = []
        built = build_circuit(qobj.experiments[0].instructions)
        self.assertEqual(expected, built)

    def test_simple_circuit(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qobj = assemble(qc, backend)
        expected = [{"gate": "h", "target": 0}]
        built = build_circuit(qobj.experiments[0].instructions)
        self.assertEqual(expected, built)

    def test_circuit_with_entangling_ops(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(2, 2)
        qc.cnot(1, 0)
        qobj = assemble(qc, backend)
        expected = [{"gate": "x", "target": 0, "controls": [1]}]
        built = build_circuit(qobj.experiments[0].instructions)
        self.assertEqual(expected, built)

    def test_multi_control(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(3, 3)
        qc.toffoli(0, 1, 2)
        qobj = assemble(qc, backend)
        expected = [{"gate": "x", "target": 2, "controls": [0, 1]}]
        built = build_circuit(qobj.experiments[0].instructions)
        self.assertEqual(expected, built)

    def test_multi_control(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(3, 3)
        qc.toffoli(0, 1, 2)
        qobj = assemble(qc, backend)
        expected = [{"gate": "x", "target": 2, "controls": [0, 1]}]
        built = build_circuit(qobj.experiments[0].instructions)
        self.assertEqual(expected, built)


class TestOutputMapper(MockCredentialsTestCase):
    def test_build_simple_output_map(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(3, 3)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qobj = assemble(qc, backend)
        expected = {0: 0, 1: 1, 2: 2}
        mapped = build_output_map(qobj)
        self.assertEqual(expected, mapped)

    def test_build_extended_output_map(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(3, 6)
        qc.measure(0, 0)
        qc.measure(1, 2)
        qc.measure(2, 5)
        qobj = assemble(qc, backend)
        expected = {0: 0, 1: 2, 2: 5}
        mapped = build_output_map(qobj)
        self.assertEqual(expected, mapped)

    def test_build_truncated_output_map(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(6, 1)
        qc.measure(4, 0)
        qobj = assemble(qc, backend)
        expected = {4: 0}
        mapped = build_output_map(qobj)
        self.assertEqual(expected, mapped)

    def test_build_scrambled_output_map(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(6, 6)
        qc.measure(0, 4)
        qc.measure(1, 3)
        qc.measure(2, 5)
        qc.measure(3, 0)
        qobj = assemble(qc, backend)
        expected = {0: 4, 1: 3, 2: 5, 3: 0}
        mapped = build_output_map(qobj)
        self.assertEqual(expected, mapped)

    def test_measure_all(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(4)
        qc.measure_all()
        qobj = assemble(qc, backend)
        expected = {0: 0, 1: 1, 2: 2, 3: 3}
        mapped = build_output_map(qobj)
        self.assertEqual(expected, mapped)

    def test_measure_active(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.measure_active()
        qobj = assemble(qc, backend)
        expected = {0: 0}
        mapped = build_output_map(qobj)
        self.assertEqual(expected, mapped)

    def test_exception_on_no_measurement(self):
        backend = IonQ.get_backend("ionq_qpu")
        qc = QuantumCircuit(1)
        qobj = assemble(qc, backend)
        self.assertRaises(ValueError, build_output_map, qobj)


class TestQobjToIonQ(MockCredentialsTestCase):
    maxDiff = None

    def test_full_circuit(self):
        backend = IonQ.get_backend("ionq_simulator")
        qc = QuantumCircuit(2, 2, name="test_name")
        qc.cnot(1, 0)
        qc.h(1)
        qc.measure(1, 0)
        qc.measure(0, 1)
        qobj = assemble(qc, backend, "test_id", shots=200)
        ionq_json = qobj_to_ionq(qobj)
        expected_metadata_header = {
            "clbit_labels": [["c", 0], ["c", 1]],
            "creg_sizes": [["c", 2]],
            "global_phase": 0,
            "memory_slots": 2,
            "n_qubits": 2,
            "name": "test_name",
            "qreg_sizes": [["q", 2]],
            "qubit_labels": [["q", 0], ["q", 1]],
        }
        expected_output_map = {"0": 1, "1": 0}
        expected_metadata = {
            "output_length": "2",
            "qobj_id": "test_id",
            "shots": "200",
        }
        expected = {
            "lang": "json",
            "target": "simulator",
            "shots": 200,
            "body": {
                "qubits": 2,
                "circuit": [
                    {"gate": "x", "controls": [1], "target": 0},
                    {"gate": "h", "target": 1},
                ],
            },
        }
        expected_json = json.dumps(expected)

        actual = json.loads(ionq_json)
        actual_metadata = actual.pop("metadata") or {}
        actual_output_map = json.loads(actual_metadata.pop("output_map") or "{}")
        actual_metadata_header = json.loads(actual_metadata.pop("header") or "{}")

        # check dict equality:
        self.assertEqual(expected, actual)
        self.assertEqual(expected_metadata, actual_metadata)
        self.assertEqual(expected_metadata_header, actual_metadata_header)
        self.assertEqual(expected_output_map, actual_output_map)
