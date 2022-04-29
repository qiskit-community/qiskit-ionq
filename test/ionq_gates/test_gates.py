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

from qiskit_ionq import GPIGate, GPI2Gate, MSGate


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, math.pi/2, math.pi, 2*math.pi])
def test_gpi_inverse(phase):
    """Tests that the inverse GPI gate is correct."""
    gate = GPIGate(phase)

    mat = numpy.array(gate)
    inv = numpy.array(gate.inverse())
    numpy.testing.assert_array_almost_equal(inv, numpy.linalg.inv(mat))

@pytest.mark.parametrize("phase", [0, 0.1, 0.4, math.pi/2, math.pi, 2*math.pi])
def test_gpi2_inverse(phase):
    """Tests that the inverse GPI2 gate is correct."""
    gate = GPI2Gate(phase)

    mat = numpy.array(gate)
    inv = numpy.array(gate.inverse())
    numpy.testing.assert_array_almost_equal(inv, numpy.linalg.inv(mat))

@pytest.mark.parametrize("phases", [(0, 1), (0.1, 1), (0.4, 1), (math.pi/2, 0), (0, math.pi), (0.1, 2*math.pi)])
def test_ms_inverse(phases):
    """Tests that the inverse MS gate is correct."""
    gate = MSGate(phases[0], phases[1])

    mat = numpy.array(gate)
    inv = numpy.array(gate.inverse())
    numpy.testing.assert_array_almost_equal(inv, numpy.linalg.inv(mat))
