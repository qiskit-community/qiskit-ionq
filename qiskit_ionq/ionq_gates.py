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

"""Native gateset for IonQ hardware.

Phase parameters (``phi``, ``phi0``, ``phi1``) are expressed in turns
(fractions of 2*pi). Interaction parameters (``theta``) are in units of pi
(0.25 = pi/4 radians). The closed-form unitaries are computed by
``ionq_core.{gpi_matrix, gpi2_matrix, ms_matrix, zz_matrix}`` so this module
and any other downstream consumer of those matrices stay in sync.
"""

from typing import Optional

import numpy as np
from ionq_core import gpi2_matrix, gpi_matrix, ms_matrix, zz_matrix
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType


def _coerce(arr: np.ndarray, dtype, copy) -> np.ndarray:
    """Match the legacy dtype/copy semantics of ``Gate.__array__``."""
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    if copy is True:
        return arr.copy()
    return arr


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

    def __array__(self, dtype=None, copy=None):
        """Return a numpy array for the GPI gate."""
        return _coerce(np.array(gpi_matrix(float(self.params[0]))), dtype, copy)


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
            \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1 & -i*e^{-i*2*\pi*\phi} \\
                -i*e^{i*2*\pi*\phi} & 1
            \end{pmatrix}
    """

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new GPI2 gate."""
        super().__init__("gpi2", 1, [phi], label=label)

    def __array__(self, dtype=None, copy=None):
        """Return a numpy array for the GPI2 gate."""
        return _coerce(np.array(gpi2_matrix(float(self.params[0]))), dtype, copy)


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

    def __array__(self, dtype=None, copy=None):
        """Return a numpy array for the MS gate."""
        phi0, phi1, theta = (float(p) for p in self.params)
        return _coerce(np.array(ms_matrix(phi0, phi1, theta)), dtype, copy)


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

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return a numpy array for the ZZ gate."""
        return _coerce(np.array(zz_matrix(float(self.params[0]))), dtype, copy)
