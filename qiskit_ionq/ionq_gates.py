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
        q_0: ┤ GPI(ϴ)├
             └───────┘
    **Matrix Representation:**
    .. math::
        GPI(\phi) =
            \begin{pmatrix}
                0 & e^{-i\phi} \\
                e^{-i\phi} & 0
            \end{pmatrix}
    """

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new GPI gate."""
        super().__init__("gpi", 1, [phi], label=label)

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
        q_0: ┤GPI2(ϴ)├
             └───────┘
    **Matrix Representation:**
    .. math::
        GPI2(\phi) =
            \begin{pmatrix}
                1 & -i*e^{-i\phi} \\
                -i*e^{i\phi} & 1
            \end{pmatrix}
    """

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new GPI2 gate."""
        super().__init__("gpi2", 1, [phi], label=label)

    def __array__(self, dtype=None):
        """Return a numpy.array for the GPI2 gate."""
        top = -1j * cmath.exp(self.params[0] * -1j)
        bot = -1j * cmath.exp(self.params[0] * 1j)
        return numpy.array([[1, top], [bot, 1]], dtype=dtype)/math.sqrt(2)


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
        MS(\phi_0, _\phi_1) q_0, q_1 =
            MS(\phi_1 - \phi_0) =
            MS(t) =
            \frac{1}{\sqrt{2}}\begin{pmatrix}
                \cos(t) & 0         & 0 & -i*\sin(t) \\
                0 & \cos(t) & -i*\sin(t) & 0 \\
                0 & -i*\sin(t) & \cos(t) & 0 \\
                -i*\sin(t) & 0 & 0 & \cos(t)
            \end{pmatrix}
    """

    def __init__(
        self,
        phi0: ParameterValueType,
        phi1: ParameterValueType,
        label: Optional[str] = None,
    ):
        """Create new MS gate."""
        super().__init__(
            "ms",
            2,
            [phi0, phi1],
            label=label,
        )

    def __array__(self, dtype=None):
        """Return a numpy.array for the MS gate."""
        tee = self.params[1] - self.params[0]
        diag = math.cos(tee)
        adiag = -1j*math.sin(tee)
        return numpy.array(
            [[diag, 0, 0, adiag], [0, diag, adiag, 0], [0, adiag, diag, 0], [adiag, 0, 0, diag]],
            dtype=dtype,
        )
