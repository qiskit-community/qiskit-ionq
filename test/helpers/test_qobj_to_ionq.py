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

"""Test the qobj_to_ionq function."""

import json

from qiskit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit_ionq_provider import IonQProvider
from qiskit_ionq_provider.helpers import build_output_map, qobj_to_ionq

from ..base import MockCredentialsTestCase

IonQ = IonQProvider()

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

        actual = json.loads(ionq_json)
        actual_metadata = actual.pop("metadata") or {}
        actual_output_map = json.loads(
            actual_metadata.pop("output_map") or "{}"
        )
        actual_metadata_header = json.loads(
            actual_metadata.pop("header") or "{}"
        )

        # check dict equality:
        self.assertEqual(expected, actual)
        self.assertEqual(expected_metadata, actual_metadata)
        self.assertEqual(expected_metadata_header, actual_metadata_header)
        self.assertEqual(expected_output_map, actual_output_map)
