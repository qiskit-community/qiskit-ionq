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

"""Tests for the IonQ's GPIGate, GPI2Gate, MSGate, ZZGate."""
# pylint: disable=redefined-outer-name

import numpy as np

import pytest

from qiskit.circuit.library import XGate, YGate, RXGate, RYGate, HGate
from qiskit_ionq import GPIGate, GPI2Gate, MSGate, ZZGate


@pytest.mark.parametrize("gate,phase", [(XGate(), 0), (YGate(), 0.25)])
def test_gpi_equivalences(gate, phase):
    """Tests equivalence of the GPI gate at specific phases."""
    gpi = GPIGate(phase)
    np.testing.assert_array_almost_equal(gate.to_matrix(), gpi.to_matrix())


@pytest.mark.parametrize(
    "gate,phase", [(RXGate(np.pi / 2), 1), (RYGate(np.pi / 2), 0.25)]
)
def test_gpi2_equivalences(gate, phase):
    """Tests equivalence of the GPI2 gate at specific phases."""
    gpi2 = GPI2Gate(phase)
    np.testing.assert_array_almost_equal(gate.to_matrix(), gpi2.to_matrix())


@pytest.mark.parametrize("gpi2_angle_1, gpi_angle, gpi2_angle_2", [(0, -0.125, 0.5)])
def test_hadamard_equivalence(gpi2_angle_1, gpi_angle, gpi2_angle_2):
    """Tests equivalence of the Hadamard gate with the GPI and GPI2 gates."""
    gpi2_1 = GPI2Gate(gpi2_angle_1)
    gpi = GPIGate(gpi_angle)
    gpi2_2 = GPI2Gate(gpi2_angle_2)
    native_hadamard = np.dot(gpi2_2, np.dot(gpi, gpi2_1))
    np.testing.assert_array_almost_equal(native_hadamard, HGate().to_matrix())


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, np.pi / 2, np.pi, 2 * np.pi])
def test_gpi_inverse(phase):
    """Tests that the GPI gate is unitary."""
    gate = GPIGate(phase)
    mat = np.array(gate)
    np.testing.assert_array_almost_equal(mat.dot(mat.conj().T), np.identity(2))


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, np.pi / 2, np.pi, 2 * np.pi])
def test_gpi2_inverse(phase):
    """Tests that the GPI2 gate is unitary."""
    gate = GPI2Gate(phase)

    mat = np.array(gate)
    np.testing.assert_array_almost_equal(mat.dot(mat.conj().T), np.identity(2))


@pytest.mark.parametrize(
    "params",
    [
        (0, 1, 0.25),
        (0.1, 1, 0.25),
        (0.4, 1, 0.25),
        (np.pi / 2, 0, 0.25),
        (0, np.pi, 0.25),
        (0.1, 2 * np.pi, 0.25),
    ],
)
def test_ms_inverse(params):
    """Tests that the MS gate is unitary."""
    gate = MSGate(params[0], params[1], params[2])

    mat = np.array(gate)
    np.testing.assert_array_almost_equal(mat.dot(mat.conj().T), np.identity(4))


@pytest.mark.parametrize(
    "angle",
    [0, 0.1, 0.4, np.pi / 2, np.pi, 2 * np.pi],
)
def test_zz_inverse(angle):
    """Tests that the ZZ gate is unitary."""
    gate = ZZGate(angle)

    mat = np.array(gate)
    np.testing.assert_array_almost_equal(mat.dot(mat.conj().T), np.identity(4))
