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
import math
import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType


class GPIGate(Gate):
    r"""Single-qubit GPI gate.
    **Circuit symbol:**
    .. parsed-literal::
             ┌───────┐
        q_0: ┤ GPI(φ)├
             └───────┘
    **Matrix Representation:**

    .. math::

       GPI(\phi) =
            \begin{pmatrix}
                0 & e^{-i*2*\pi*\phi} \\
                e^{i*2*\pi*\phi} & 0
            \end{pmatrix}
    """

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new GPI gate."""
        super().__init__("gpi", 1, [phi], label=label)

    def __array__(self, dtype=None):
        """Return a numpy.array for the GPI gate."""
        top = cmath.exp(-self.params[0] * 2 * math.pi * 1j)
        bot = cmath.exp(self.params[0] * 2 * math.pi * 1j)
        return numpy.array([[0, top], [bot, 0]], dtype=dtype)


class GPI2Gate(Gate):
    r"""Single-qubit GPI2 gate.
    **Circuit symbol:**
    .. parsed-literal::
             ┌───────┐
        q_0: ┤GPI2(φ)├
             └───────┘
    **Matrix Representation:**

    .. math::

        GPI2(\phi) =
            \begin{pmatrix}
                1 & -i*e^{-i*2*\pi*\phi} \\
                -i*e^{i*2*\pi*\phi} & 1
            \end{pmatrix}
    """

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new GPI2 gate."""
        super().__init__("gpi2", 1, [phi], label=label)

    def __array__(self, dtype=None):
        """Return a numpy.array for the GPI2 gate."""
        top = -1j * cmath.exp(self.params[0] * 2 * math.pi * -1j)
        bot = -1j * cmath.exp(self.params[0] * 2 * math.pi * 1j)
        return numpy.array([[1, top], [bot, 1]], dtype=dtype) / math.sqrt(2)


class MSGate(Gate):
    r"""Entangling 2-Qubit MS gate.
    **Circuit symbol:**
    .. parsed-literal::
              _______
        q_0: ┤       ├-
             |MS(ϴ,0)|
        q_1: ┤       ├-
             └───────┘
    **Matrix representation:**

    .. math::

       MS(\phi_0, \phi_1, \theta) =
            \begin{pmatrix}
                cos(\theta*\pi) & 0 & 0 & -i*e^{-i*2*\pi(\phi_0+\phi_1)}*sin(\theta*\pi) \\
                0 & cos(\theta*\pi) & -i*e^{i*2*\pi(\phi_0-\phi_1)}*sin(\theta*\pi) & 0 \\
                0 & -i*e^{-i*2*\pi(\phi_0-\phi_1)}*sin(\theta*\pi) & cos(\theta*\pi) & 0 \\
                -i*e^{i*2*\pi(\phi_0+\phi_1)}*sin(\theta*\pi) & 0 & 0 & cos(\theta*\pi)
            \end{pmatrix}
    """

    def __init__(
        self,
        phi0: ParameterValueType,
        phi1: ParameterValueType,
        theta: Optional[ParameterValueType] = 0.25,
        label: Optional[str] = None,
    ):
        """Create new MS gate."""
        super().__init__(
            "ms",
            2,
            [phi0, phi1, theta],
            label=label,
        )

    def __array__(self, dtype=None):
        """Return a numpy.array for the MS gate."""
        phi0 = self.params[0]
        phi1 = self.params[1]
        theta = self.params[2]
        diag = numpy.cos(math.pi * theta)
        sin = numpy.sin(math.pi * theta)

        return numpy.array(
            [
                [diag, 0, 0, sin * -1j * cmath.exp(-1j * 2 * math.pi * (phi0 + phi1))],
                [0, diag, sin * -1j * cmath.exp(1j * 2 * math.pi * (phi0 - phi1)), 0],
                [0, sin * -1j * cmath.exp(-1j * 2 * math.pi * (phi0 - phi1)), diag, 0],
                [sin * -1j * cmath.exp(1j * 2 * math.pi * (phi0 + phi1)), 0, 0, diag],
            ],
            dtype=dtype,
        )


class ZZGate(Gate):
    r"""Two-qubit ZZ-rotation gate.
    **Circuit Symbol:**
    .. parsed-literal::
        q_0: ───■────
                │zz(θ)
        q_1: ───■────
    **Matrix Representation:**

    .. math::

        ZZ(\theta) =
            \begin{pmatrix}
                e^{-i \theta*\pi} & 0 & 0 & 0 \\
                0 & e^{i \theta*\pi} & 0 & 0 \\
                0 & 0 & e^{i \theta*\pi} & 0 \\
                0 & 0 & 0 & e^{-i \theta\*\pi}
            \end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new ZZ gate."""
        super().__init__("zz", 2, [theta], label=label)

    def __array__(self, dtype=None) -> numpy.ndarray:
        """Return a numpy.array for the ZZ gate."""
        itheta2 = 1j * float(self.params[0]) * math.pi
        return numpy.array(
            [
                [cmath.exp(-itheta2), 0, 0, 0],
                [0, cmath.exp(itheta2), 0, 0],
                [0, 0, cmath.exp(itheta2), 0],
                [0, 0, 0, cmath.exp(-itheta2)],
            ],
            dtype=dtype,
        )
