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

"""Tests for the IonQ's GPIGate, GPI2Gate, MSGate."""
# pylint: disable=redefined-outer-name

import math
import numpy

import pytest

from qiskit.circuit.library import XGate, YGate, RXGate, RYGate
from qiskit_ionq import GPIGate, GPI2Gate, MSGate

@pytest.mark.parametrize("gate,phase", [(XGate(), 0), (YGate(), 0.25)])
def test_gpi_equivalences(gate, phase):
    """Tests equivalence of the GPI gate at specific phases."""
    gpi = GPIGate(phase)
    numpy.testing.assert_array_almost_equal(numpy.array(gate), numpy.array(gpi))

@pytest.mark.parametrize("gate,phase", [(RXGate(math.pi/2), 1), (RYGate(math.pi/2), 0.25)])
def test_gpi2_equivalences(gate, phase):
    """Tests equivalence of the GPI2 gate at specific phases."""
    gpi = GPI2Gate(phase)
    numpy.testing.assert_array_almost_equal(numpy.array(gate), numpy.array(gpi))

@pytest.mark.parametrize("phase", [0, 0.1, 0.4, math.pi / 2, math.pi, 2 * math.pi])
def test_gpi_inverse(phase):
    """Tests that the GPI gate is unitary."""
    gate = GPIGate(phase)
    mat = numpy.array(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(2))


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, math.pi / 2, math.pi, 2 * math.pi])
def test_gpi2_inverse(phase):
    """Tests that the GPI2 gate is unitary."""
    gate = GPI2Gate(phase)

    mat = numpy.array(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(2))


@pytest.mark.parametrize(
    "params", [
        (0, 1, 0.25),
        (0.1, 1, 0.25),
        (0.4, 1, 0.25),
        (math.pi / 2, 0, 0.25),
        (0, math.pi, 0.25),
        (0.1, 2 * math.pi, 0.25)],
)
def test_ms_inverse(params):
    """Tests that the MS gate is unitary."""
    gate = MSGate(params[0], params[1], params[2])

    mat = numpy.array(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(4))
