# -*- coding: utf-8 -*-
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

"""Test the format_results function"""

from qiskit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit_ionq_provider import IonQProvider
from qiskit_ionq_provider.ionq_client import IonQClient
from qiskit_ionq_provider.ionq_job import IonQJob
from qiskit.qobj import QobjExperimentHeader

from ..base import MockCredentialsTestCase

IonQ = IonQProvider()

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
