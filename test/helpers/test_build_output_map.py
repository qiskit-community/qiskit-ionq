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

from qiskit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit.providers.ionq import IonQ
from qiskit.providers.ionq.helpers import build_output_map

from ..base import MockCredentialsTestCase


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
