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

# Copyright 2021 IonQ, Inc. (www.ionq.com)
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

"""Native gateset for IonQ hardware."""

from typing import Optional
import cmath
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType


class GPIGate(Gate):
    r"""Single-qubit GPI gate.
    **Circuit symbol:**
    .. parsed-literal::
             ┌───────┐
        q_0: ┤ GPI(ϴ)├
             └───────┘
    **Matrix Representation:**
    .. math::
        \newcommand{\th}{\frac{\theta}{2}}
        GPI(\phi) =
            \begin{pmatrix}
                0 & e^{-i\phi} \\
                e^{-i\phi} & 0
            \end{pmatrix}
    """

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new GPI gate."""
        super().__init__("gpi", 1, [phi], label=label)

    def inverse(self):
        r"""Return inverted GPI gate.
        :math:`GPI(\lambda){\phi} = GPI(\phi)`
        """
        return GPIGate(self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the GPI gate."""
        top = cmath.exp(self.params[0] * 1j)
        bot = cmath.exp(-self.params[0] * 1j)
        return numpy.array([[0, top], [bot, 0]], dtype=dtype)


class GPI2Gate(Gate):
    r"""Single-qubit GPI2 gate.
    **Circuit symbol:**
    .. parsed-literal::
             ┌───────┐
        q_0: ┤ GPI2(ϴ)├
             └───────┘
    **Matrix Representation:**
    .. math::
        \newcommand{\th}{\frac{\theta}{2}}
        GPI2(\phi) =
            \begin{pmatrix}
                1 & -i*e^{-i\phi} \\
                -i*e^{i\phi} & 1
            \end{pmatrix}
    """

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new GPI2 gate."""
        super().__init__("gpi2", 1, [phi], label=label)

    def inverse(self):
        r"""Return inverted GPI2 gate.
        :math:`GPI2(\lambda){\phi} = GPI2(-\phi)`
        """
        return GPI2Gate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the GPI gate."""
        top = -1j * cmath.exp(-self.params[0] * 1j)
        bot = -1j * cmath.exp(self.params[0] * 1j)
        return numpy.array([[1, top], [bot, 1]], dtype=dtype)


class MSGate(ControlledGate):
    r"""Entangling 2-Qubit MS gate.
    **Circuit symbol:**
    .. parsed-literal::
        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ MS(ϴ) ├
             └───────┘
    **Matrix representation:**
    .. math::
        \newcommand{\th}{\frac{\theta}{2}}
        MS q_0, q_1 =
            \begin{pmatrix}
                1 & 0         & 0 & -i \\
                0 & 1 & -i & 0 \\
                0 & -i         & 1 & 0 \\
                -i & 0 & 0 & 1
            \end{pmatrix}
    """

    def __init__(
        self,
        label: Optional[str] = None,
    ):
        """Create new MS gate."""
        super().__init__(
            "ms",
            1,
            [],
            num_ctrl_qubits=1,
            label=label,
        )

    def __array__(self, dtype=None):
        """Return a numpy.array for the MS gate."""
        return numpy.array(
            [[1, 0, 0, -1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [-1j, 0, 0, 0]],
            dtype=dtype,
        )
