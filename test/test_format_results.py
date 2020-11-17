import os
import unittest

from qiskit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit.providers.ionq import IonQ
from qiskit.providers.ionq.ionq_client import IonQClient
from qiskit.providers.ionq.ionq_job import IonQJob
from qiskit.qobj import QobjExperimentHeader

from .base import MockCredentialsTestCase


class StubbedClient(IonQClient):
    def __init__(self):
        super().__init__(self)

    def retrieve_job(self, job_id):
        return {
            "status": "completed",
            "predicted_execution_time": 4,
            "metadata": {
                "shots": "1234",
                "qobj_id": "test_qobj_id",
                "output_length": "2",
                "output_map": '{"0": 1, "1": 0}',
                "header": '{"qubit_labels": [["q", 0], ["q", 1]], "n_qubits": 2, "qreg_sizes": [["q", 2]], "clbit_labels": [["c", 0], ["c", 1]], "memory_slots": 2, "creg_sizes": [["c", 2]], "name": "test-circuit", "global_phase": 0}',
            },
            "execution_time": 8,
            "qubits": 2,
            "type": "circuit",
            "request": 1600000000,
            "start": 1600000001,
            "response": 1600000002,
            "data": {"histogram": {"0": 0.5, "2": 0.499999}},
            "target": "qpu",
            "id": "test_id",
        }


class TestResultsFormatter(MockCredentialsTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        backend = IonQ.get_backend("ionq_qpu")
        client = StubbedClient()
        job = IonQJob(backend, "test_id", client)
        cls.formatted = job.result()

    def test_results_meta(self):
        self.assertEqual(self.formatted.backend_name, "ionq_qpu")
        self.assertEqual(self.formatted.backend_version, "0.0.1")
        self.assertEqual(self.formatted.qobj_id, "test_qobj_id")
        self.assertEqual(self.formatted.job_id, "test_id")
        self.assertEqual(self.formatted.success, True)

    def test_counts(self):
        counts = self.formatted.get_counts()
        print(self.formatted.get_counts())
        self.assertEqual({"00": 617, "01": 617}, counts)

    def test_additional_results_details(self):
        results = self.formatted.results[0]
        self.assertEqual(results.shots, "1234")
        self.assertEqual(results.success, True)
        self.assertEqual(
            results.header,
            QobjExperimentHeader.from_dict(
                {
                    "qubit_labels": [["q", 0], ["q", 1]],
                    "n_qubits": 2,
                    "qreg_sizes": [["q", 2]],
                    "clbit_labels": [["c", 0], ["c", 1]],
                    "memory_slots": 2,
                    "creg_sizes": [["c", 2]],
                    "name": "test-circuit",
                    "global_phase": 0,
                }
            ),
        )
