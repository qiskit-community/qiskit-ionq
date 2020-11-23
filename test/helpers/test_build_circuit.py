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
from qiskit_ionq_provider.helpers import build_circuit

from ..base import MockCredentialsTestCase

IonQ = IonQProvider()

class TestCircuitBuilder(MockCredentialsTestCase):
    """Test the `build_circuit` function."""

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
